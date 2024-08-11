import numpy as np
import dklib.thermal_activation
import pickle
import dklib.plothelp
import dklib.looputil
import pickle
import matplotlib.pyplot as plt
import dklib.running_hist
import dklib.hist_weighted
import sys
from mpi4py import MPI
from dklib.mpi_utils import is_root, printmpi, printroot, get_rank, unifyarr, unifyvar
from dklib.histnd import histNd
import dklib.histnd
import dklib.arrhenius_loading_08 as arrhenius_loading

import dklib.mf_epm as mf_epm
import dklib.mf_avalanche as mf_avalanche

import sim_params_01 as sim_params

DO_NOT_SAVE = False
debug = False
debug_level = 0
debug_weird_failure = False


dsig_temp = None


def apply_kicks(mf, failingSites):
    global dsig_temp
    if dsig_temp is None:
        dsig_temp = np.zeros(mf.N)

    for ind in failingSites:
        kickSource.compute_kicks(ind, rng, dsig_temp)

        if is_root():
            mf.sigma_th[ind] = rng.weibull(
                renewal_k,
            )
            sig = mf.sigma[ind]
            mf.sigma_res += sig * dsig_temp
            cumulative_eigenstrain[ind] += -sig * dsig_temp[ind]
    if is_root():
        mf.tlast[failingSites] = mf.t
        mf.tsolid[failingSites] = mf.fluidization_time()
        mf.compute_x()


# ===== read script input:
import argparse

parser = argparse.ArgumentParser(
    description="runs thermally activated simulations of a mean-field EPM"
)
parser.add_argument(
    "-runno",
    dest="runno",
    metavar="m",
    help="run number, iterating through the parameter array",
    type=int,
)
parser.add_argument(
    "-data_folder",
    dest="data_folder",
    help="Directory to output the data",
    type=str,
    default="",
)
parser.add_argument("-L", dest="L", metavar="L", help="Linear dimension", type=int)
parser.add_argument(
    "-src",
    dest="loadsrc",
    metavar="src",
    help="file containing save file state from which to load",
    type=str,
    default="",
)
parser.add_argument(
    "--use_final_state",
    dest="use_final_state",
    help='should we check the source for a subdictionary "[final_state]"?',
    action="store_true",
)
parser.add_argument(
    "-renewal_k",
    dest="renewal_k",
    default=sim_params.RENEWAL_K,
    help="Which k exponent should we choose for renewing sites?",
    type=float,
)
parser.add_argument(
    "-anneal_k",
    dest="anneal_k",
    default=sim_params.ANNEAL_K,
    help="Which k exponent should we choose for the initial annealing of sites?",
    type=float,
)
parser.add_argument(
    "--arrhenius",
    dest="arrhenius",
    default=True,
    help="Do we use the arrhenius activation rate to pre-empt loading?",
    action="store_true",
)
parser.add_argument(
    "-kicksource",
    dest="kicksource",
    help='What kind of kick source do we use with arrhenius activation? Choose from "2d", "shuffled", "gaussian"',
    type=str,
    default="2d",
)
parser.set_defaults(use_final_state=False)
args = parser.parse_args()


kickSourceType = args.kicksource
DATA_FOLDER = args.data_folder
newSimulation = len(args.loadsrc) == 0
if newSimulation:
    repno = args.runno
    # setting up the simulation:
    L = args.L
    v, beta, alpha = sim_params.get_params(repno, L)
    gaussianMagnitude = 0.5
    params = mf_epm.ArrheniusThermalParams(v, beta, alpha, sim_params.TAU_R)
    use_arrhenius = True

    rng_seed = 1337 + 120120 * repno + L * 367
    rng = np.random.default_rng(rng_seed)
    mf = mf_epm.MFEPM(L**2, params)
    # initiate the mean-field:
    anneal_k = args.anneal_k
    mf.sigma_th = rng.weibull(anneal_k, size=mf.N)
    # building the renewal distributions:
    renewal_k = args.renewal_k
    printroot("starting with seed: ", rng_seed)
else:
    with open(args.loadsrc, "rb") as fh:
        dic = pickle.load(fh)
        L = dic["L"]
        if args.use_final_state:
            dic = dic["final_state"]
        repno = dic["repno"]
        mf = dic["mf"]
        rng = dic["rng"]
        params = dic["params"]
        if "use_arrhenius" in dic.keys():
            use_arrhenius = dic["use_arrhenius"]
        else:
            use_arrhenius = args.arrhenius

        if not use_arrhenius:
            D, v = params.D, params.v
        else:
            v, beta, alpha = params.v, params.beta, params.alpha

        # setting the kick source up.
        if "use_shuffled_kernel" in dic.keys():
            kickSourceType = "shuffled"

        if "kickSourceType" in dic.keys():
            kickSourceType = dic["kickSourceType"]
        if "gaussianMagnitude" in dic.keys():
            gaussianMagnitude = dic["gaussianMagnitude"]

        # setting the renewal and annealing k.
        if "renewal_k" in dic.keys():
            renewal_k = dic["renewal_k"]
        else:
            renewal_k = args.renewal_k
        if "anneal_k" in dic.keys():
            anneal_k = dic["anneal_k"]
        else:
            anneal_k = args.anneal_k


# printing what the simulation is starting with.
if not use_arrhenius:
    printroot("starting: ", L, D, v, "ks: ", (anneal_k, renewal_k))
else:
    printroot("starting: ", L, beta, v, alpha, "ks: ", (anneal_k, renewal_k))

kickSource = None
import dklib.kicksources_03 as kicksources

if kickSourceType == "2d":
    kickSource = kicksources.fem_kernel(L)
    print("built kicksource, using 2d")
elif kickSourceType == "shuffled":
    print("built kicksource, using shuffled")
    kickSource = kicksources.shuffled_kernel(L)
elif kickSourceType == "gaussian":
    kickSource = kicksources.gaussian_kernel(L, gaussianMagnitude)
    print("built kicksource, using gaussian")
else:
    print("was not supplied with a valid kick source")

sim_ident_string = "%4.4d_%4.4d_%.2f_%.2f" % (repno, L, anneal_k, renewal_k)
sim_ident_string += "_" + kickSourceType

import dklib.epm_plaquette as epm

import shutil
import os


def save_output():
    if is_root():
        output_name = DATA_FOLDER + "simulation_" + sim_ident_string + ".tmp"
        output_name_final = DATA_FOLDER + "simulation_" + sim_ident_string + ".dat"
        print("attempting saving to: ", output_name)
        output_dic = {}
        output_dic["avs"] = mf_avalanche.avalanche.list2np(avs)
        output_dic["L"] = L
        output_dic["renewal_k"] = renewal_k
        output_dic["anneal_k"] = anneal_k
        output_dic["rep_params"] = sim_params.get_params(repno, L)
        output_dic["sts"] = mf_avalanche.STRecord.list2np(sts)
        output_dic["init_fail_sites"] = np.array(init_fail_sites)
        output_dic["params"] = mf.params
        output_dic["px"] = px_av_sample
        output_dic["px_history"] = px_av_sample_list
        output_dic["pdsig_ps"] = pdsig_ps_sample
        output_dic["pdsig_ps_history"] = pdsig_ps_sample_list
        output_dic["psig"] = psig_sample
        output_dic["psig_history"] = psig_sample_list
        output_dic["psig_fail"] = psig_fail_sample
        output_dic["psig_fail_history"] = psig_fail_sample_list
        output_dic["cumulative_eigenstrain"] = cumulative_eigenstrain
        output_dic["cumulative_eigenstrain_history"] = cumulative_eigenstrain_list
        output_dic["avalanche_site_records"] = avalanche_site_records
        output_dic["px_t_sampled"] = px_t_sample
        output_dic["repno"] = repno
        output_dic["final_state"] = {
            "mf": mf,
            "rng": rng,
            "params": params,
            "repno": repno,
            "L": L,
        }
        output_dic["kickSourceType"] = kickSourceType
        output_dic["gaussianMagnitude"] = gaussianMagnitude
        output_dic["use_arrhenius"] = use_arrhenius

        if not DO_NOT_SAVE:
            if os.path.exists(output_name_final):
                shutil.move(output_name_final, output_name_final + ".old")
            with open(output_name, "wb") as fh:
                pickle.dump(output_dic, fh)
            shutil.move(output_name, output_name_final)


# ========= main loop:
# data recording
avs = []
sts = (
    []
)  # Wouldn't it be nice to record the failure time of each ST, the stress at the moment of failure, and how much residual stress there is in the system?
init_fail_sites = []  # measure (stress,stress_threshold,twait)


# some timing varriables
cumulative_time_kicking = 0.0
cumulative_sync_time = 0.0
cumulative_time_brownian = 0.0
cumulative_loop_decision_time = 0.0
cum_inner_loop_time = 0.0

# px histogram:
px_av_sample = dklib.running_hist.hist(sim_params.PX_BINS)
px_t_sample = dklib.hist_weighted.hist(sim_params.PX_BINS)
# px histograms, as a time-series.
histogram_cumulative_stress_lower_bound = 0.0
px_av_sample_list = []

# dsig_avalanche_size multihistogram.
pdsig_ps_sample = dklib.histnd.histNd([sim_params.DSIG_BINS, sim_params.AVSIZE_BINS])
pdsig_ps_sample_list = []

# psig histogram:
psig_sample = dklib.running_hist.hist(sim_params.SIG_BINS)
psig_sample_list = []

# psig_fail_sample:
psig_fail_sample = dklib.running_hist.hist(sim_params.SIG_BINS)
psig_fail_sample_list = []

# ========= Map-type variables:
# cumulative eigenstrain:
cumulative_eigenstrain = np.zeros(L * L)
cumulative_eigenstrain_list = []
map_cumulative_stress_lower_bound = 0.0

# ===== Failing site indices:
avalanche_site_records = []
current_avalanche_site_record = []

# per-av tracking variables:
av_solid_sites = np.ones(mf.N, bool)
av_sig1 = np.zeros(mf.N)
av_sig2 = np.zeros(mf.N)
av_dsig = np.zeros(mf.N)

# initiating the loop:
newAv = mf_avalanche.avalanche()
num_sts = 0  # number of sts within an avalnahce
av_num = 0
bar_avs = dklib.looputil.progressBar(10)
bar_time = dklib.looputil.progressBar(10)
bar_strain = dklib.looputil.progressBar(10)
bars_sts = dklib.looputil.progressBar(10)
import time

tstart = time.time()
stateOutputTimeInd = 0
lastSaveTime = tstart
while True:
    bar_avs.update(av_num, sim_params.MAX_AVS, prepend="avalanche count: ")
    bar_strain.update(
        -mf.t * mf.params.v, sim_params.MAX_STRAIN, prepend="strain progress: "
    )
    bar_time.update(
        time.time() - tstart,
        sim_params.MAX_SIMULATION_WTIME_L[L],
        prepend="wall time progress: ",
    )

    should_break = False
    # various checks to see if we should stop or if we should save the system state.
    if is_root():
        # checking if we've exceeded the maximum number of avalanches.
        if av_num >= sim_params.MAX_AVS:
            printroot("hit max avalanches")
            should_break = True
        # checking to see if we've arrived at maximum strain.
        if mf.t * -mf.params.v > sim_params.MAX_STRAIN:
            printroot(
                "hit max strain: ", mf.t, mf.t * -mf.params.v, sim_params.MAX_STRAIN
            )
            should_break = True

        # checking if we should save the intermediate state of the system.
        if (
            stateOutputTimeInd < np.size(sim_params.SAVE_OUTPUT_TIMES)
            and time.time() - tstart > sim_params.SAVE_OUTPUT_TIMES[stateOutputTimeInd]
        ):
            if not DO_NOT_SAVE:
                dic = {"mf": mf, "rng": rng, "params": params, "repno": repno, "L": L}
                fname = DATA_FOLDER + "state_" + sim_ident_string + ".dat"
                if os.path.exists(fname):
                    shutil.move(fname, fname + ".old")
                with open(fname + ".tmp", "wb") as fh:
                    pickle.dump(dic, fh)
                shutil.move(fname + ".tmp", fname)
            print("saving system state: ", (time.time() - tstart) / 60.0)
            stateOutputTimeInd += 1

        # checking to see if we should save the main file.
        if time.time() - lastSaveTime > sim_params.AUTOSAVE_INTERVAL:
            printroot("autosaving!", time.time() - lastSaveTime)
            lastSaveTime = time.time()
            # save output:
            save_output()

        # checking if we have exceeded the simulation time
        if time.time() - tstart > sim_params.MAX_SIMULATION_WTIME_L[L]:
            printroot("hit max wall time")
            should_break = True

        # checking if, when recording STs, we've exceeded the number we should have.
        if sim_params.RECORD_STS:
            if len(sts) > sim_params.MAX_TOTAL_STS:
                printroot("hit maximum number of sts, stopping loading")
                should_break = True
    t1 = time.time()
    should_break = unifyvar(should_break, bool, MPI.BOOL)
    t2 = time.time()
    cumulative_sync_time += t2 - t1
    if should_break:
        break

    # during each avalanche, we need to load to failure repeatedly, until the loading to failure is too long, or the avalanche is too big:

    # push the system until something fails:
    failing_sites = []
    current_avalanche_site_record = []
    t1 = time.time()
    if is_root():
        if not use_arrhenius:
            assert False
        else:
            failing_sites, t_load = arrhenius_loading.load_to_fail_inter_avalanche(
                mf,
                rng,
                sim_params.INTER_AVALANCHE_NUMERICAL_TIME,
                debug_level=debug_level,
            )
            printroot("done loading to fail, inter avalanche!", failing_sites, t_load)
        av_sig1[:] = mf.sigma[:]
        av_solid_sites[:] = True
        av_solid_sites[failing_sites] = False
        psig_fail_sample.addDat(mf.sigma[failing_sites], mf.t * -mf.params.v)
        if sim_params.RECORD_ST_INDS:
            current_avalanche_site_record += [
                (
                    mf.t,
                    siteind,
                    mf.sigma[siteind],
                    mf.sigma_th[siteind] - np.abs(mf.sigma[siteind]),
                )
                for siteind in failing_sites
            ]
            # current_avalanche_site_record.append((mf.t, failing_sites))
        if sim_params.RECORD_INIT_FAILING_SITES:
            fsite_ind = failing_sites[0]
            init_fail_sites.append(
                (mf.sigma[fsite_ind], mf.sigma_th[fsite_ind], t_load)
            )
    t2 = time.time()
    cumulative_time_brownian += t2 - t1
    t1 = time.time()
    failing_sites = unifyarr(failing_sites, np.int64, MPI.LONG)
    t2 = time.time()
    cumulative_sync_time += t2 - t1

    if debug_level > 0:
        print(av_num, "init failing sites: ", failing_sites)
    # now, apply kicks at failing sites:
    t1 = time.time()
    apply_kicks(mf, failing_sites)
    t2 = time.time()
    cumulative_time_kicking += t2 - t1

    # begin a new avalanche
    num_sts = failing_sites.size
    av_num += 1
    # sampling per site data:
    if is_root():
        newAv = mf_avalanche.avalanche()
        newAv.update(mf, 0, num_sts)
        px_t_sample.addDat(mf.x, weight=np.full(mf.x.shape, t_load))
    # debugging info related to starting avalanches
    if debug_level > 0:
        print("started avalanche: ", newAv.values[0, :])
    if failing_sites.size == 0:
        # somehow had nothing fail
        printmpi("Error: nothing failed!", num_sts)

    while True:
        failing_sites = []
        t1 = time.time()
        if is_root():
            if not use_arrhenius:
                failing_sites, t_load = load_to_fail(
                    mf, rng, mf.params.tau * sim_params.AVALANCHE_WINDOW_IN_TAUS
                )
            else:
                failing_sites, t_load = arrhenius_loading.load_to_fail_numerical(
                    mf,
                    rng,
                    mf.params.tau * sim_params.AVALANCHE_WINDOW_IN_TAUS,
                    debug_level=debug_level,
                )
        t2 = time.time()
        cumulative_time_brownian += t2 - t1
        t1 = time.time()
        failing_sites = unifyarr(failing_sites, np.int64, MPI.LONG)
        t2 = time.time()
        cumulative_sync_time += t2 - t1
        # now, apply kicks at failing sites:
        t1 = time.time()
        apply_kicks(mf, failing_sites)
        t2 = time.time()
        cumulative_time_kicking += t2 - t1

        # making measurements.
        should_break = False
        t1_decision = time.time()
        # compute the cumulatives stress:
        cumulative_stress = mf.t * -mf.params.v
        if is_root():
            # sampling per site data:
            px_t_sample.addDat(mf.x, weight=np.full(mf.x.shape, t_load))

            # saving the ST related data:
            if failing_sites.size > 0:
                # avalanche continues:
                if sim_params.RECORD_STS:
                    sts.append(mf_avalanche.STRecord(mf))
                    bars_sts.update(
                        len(sts),
                        sim_params.MAX_TOTAL_STS,
                        prepend="number of sts progress: ",
                    )
                    if len(sts) > sim_params.MAX_TOTAL_STS:
                        printroot("exceeded total number of STs.")
                        should_break = True
                if sim_params.RECORD_ST_INDS:
                    current_avalanche_site_record += [
                        (
                            mf.t,
                            siteind,
                            mf.sigma[siteind],
                            mf.sigma_th[siteind] - np.abs(mf.sigma[siteind]),
                        )
                        for siteind in failing_sites
                    ]
                    # current_avalanche_site_record.append((mf.t,failing_sites))
                num_sts += failing_sites.size
                av_solid_sites[failing_sites] = False
                # recording the failing site stress:
                psig_fail_sample.addDat(mf.sigma[failing_sites], cumulative_stress)
                # should we break because the simulation is too large ?
                if num_sts > sim_params.MAX_AV_SIZE:
                    printmpi("we have exceeded maximum avalanche size!")
                    should_break = True
                if debug_level > 0:
                    print(num_sts, failing_sites.size, t_load, t_load / (mf.params.tau))
                if num_sts % 100 == 0:
                    printroot(
                        "load: ",
                        mf.t * -mf.params.v,
                        "total sts: ",
                        len(sts),
                        "stress: ",
                        np.mean(mf.sigma),
                    )

            # should we terminate early, if we are running too long?
            if (
                time.time() - tstart
                > sim_params.MAX_AVALANCHE_GRACE_WTIME
                + sim_params.MAX_SIMULATION_WTIME_L[L]
            ):
                printroot(
                    "we have exceeded maximum wall time AND the grace time allowed to an avalanche."
                )
                should_break = True

            # Terminating the avalanche, either because it naturally terminated, or because it is running too long.
            if failing_sites.size == 0 or should_break:
                # no more avalanche continuation.
                should_break = True
                # avalanche end logic:
                newAv.update(mf, 1, num_sts)
                avs.append(newAv)

                print(
                    "Finished av.: ",
                    av_num,
                    "cumulative stress",
                    cumulative_stress,
                    "stress: %.6g"
                    % (newAv.values[1, mf_avalanche.avalanche.SIG_INDEX]),
                    "sites: %.6g"
                    % (newAv.values[1, mf_avalanche.avalanche.ST_COUNT_INDEX]),
                    "drop size: %.6g"
                    % (
                        newAv.values[1, mf_avalanche.avalanche.SIG_INDEX]
                        - newAv.values[0, mf_avalanche.avalanche.SIG_INDEX]
                    ),
                )
                if (
                    cumulative_stress - histogram_cumulative_stress_lower_bound
                    > sim_params.HISTOGRAM_STRESS_INTERVAL
                ):
                    histogram_cumulative_stress_lower_bound = cumulative_stress
                    px_av_sample_list.append(px_av_sample)
                    px_av_sample = dklib.running_hist.hist(sim_params.PX_BINS)
                    pdsig_ps_sample_list.append(pdsig_ps_sample)
                    pdsig_ps_sample = dklib.histnd.histNd(
                        [sim_params.DSIG_BINS, sim_params.AVSIZE_BINS]
                    )
                    psig_sample_list.append(psig_sample)
                    psig_sample = dklib.running_hist.hist(sim_params.SIG_BINS)
                    psig_fail_sample_list.append(psig_fail_sample)
                    psig_fail_sample = dklib.running_hist.hist(sim_params.SIG_BINS)

                if (
                    cumulative_stress - map_cumulative_stress_lower_bound
                    > sim_params.MAP_STRESS_INTERVAL
                ):
                    map_cumulative_stress_lower_bound = cumulative_stress
                    cumulative_eigenstrain_list.append(cumulative_eigenstrain)
                    cumulative_eigenstrain = np.zeros(L * L)

                if sim_params.RECORD_ST_INDS:
                    avalanche_site_records.append(current_avalanche_site_record)

                px_av_sample.addDat(mf.x, cumulative_stress)

                # computing stress change throughout avalanhce:
                av_sig2[:] = mf.sigma[:]
                np.subtract(av_sig2, av_sig1, out=av_dsig)

                psig_sample.addDat(av_sig2, cumulative_stress)

                non_failing_shape = np.ones(np.sum(av_solid_sites))
                avSize = (
                    -(
                        newAv.values[1, mf_avalanche.avalanche.SIG_INDEX]
                        - newAv.values[0, mf_avalanche.avalanche.SIG_INDEX]
                    )
                    * mf.N
                )
                # print('new av size: ' ,avSize)
                # print('bins: ' ,pdsig_ps_sample.binedges)
                pdsig_ps_sample.addData(
                    [av_dsig[av_solid_sites], non_failing_shape * avSize]
                )

        t1 = time.time()
        should_break = unifyvar(should_break, bool, MPI.BOOL)
        t2 = time.time()
        cumulative_sync_time += t2 - t1
        t2_decision = time.time()
        cumulative_loop_decision_time += t2_decision - t1_decision
        if should_break:
            break


# save output:
print("exited main loop.")
save_output()

print("finished everything! ", time.time() - tstart)
printmpi(
    "cumulative timers:",
    cumulative_time_brownian,
    cumulative_time_kicking,
    cumulative_sync_time,
    cumulative_loop_decision_time,
)
