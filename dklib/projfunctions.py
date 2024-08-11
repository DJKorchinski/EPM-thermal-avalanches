from dolfin import *

class Mat_cache_dict(dict):
    """Items in dictionary are matrices stored for efficient reuse.
    When the bilinear form of the mass matrix is not found, assemble and stor in cache.
    """
    def __missing__(self, key): 
        form, bcs = key
        A = assemble(form)   
        sol = KrylovSolver()
        #sol.parameters["preconditioner"]["structure"] = "same"
        for bc in bcs:
            bc.apply(A)
        
        self[key] = A, sol
        return self[key]
        
def signature(V):
    '''Compute signature = nested list representation, of the function space.
    S, scalar space, --> 0 (number of subspaces)
    V, vector space in 3D, --> 3 (number or subspaces)
    T, tensor space in 3D, --> 9 (number of subspaces)
    M = [S, V, T] --> [0, 3, 9]
    N = [M, V] --> [[0, 3, 9], 3]'''
    n_sub = V.num_sub_spaces()
    if n_sub == 0:
        return n_sub
    else:
        # Catch vector and tensor spaces
        if sum(V.sub(i).num_sub_spaces() for i in range(n_sub)) == 0:
            return n_sub
        # Proper mixed space
        else:
            n_subs = [0] * n_sub
            for i in range(n_sub):
                Vi = V.sub(i)
                n_subs[i] = signature(Vi)
            return n_subs

def mixedproject(f, Fdest,
        solver_type="cg",
        preconditioner_type="default",
        form_compiler_parameters=None):

    Udest = Function(Fdest)
    signature_ = signature(Fdest)
    L = Function(Fdest)
    assemble(inner(f, TestFunction(Fdest))*dx, tensor=L.vector())
    
    A_cache = Mat_cache_dict()

    def assemble_matrix(form, bcs=[]):
        """Assemble matrix using cache register.
        """
        assert Form(form).rank() == 2
        return A_cache[(form, tuple(bcs))]
    
    def recproj(fi, Vi, Udesti):
        signature_ = signature(Vi)
        if type(signature_) is int:
            if signature_ == 0:
                Vj = Vi.collapse()
                mass = inner(TrialFunction(Vj), TestFunction(Vj))*dx
                A, sol = assemble_matrix(mass)            
                f0 = Function(Vj)
                b = Function(Vj)
                assign(b, fi)
                sol.solve(A, f0.vector(), b.vector())
                assign(Udesti, f0)
            else:
                Vj = Vi.sub(0).collapse()
                mass = inner(TrialFunction(Vj), TestFunction(Vj))*dx
                A, sol = assemble_matrix(mass)            
                f0 = Function(Vj)
                b = Function(Vj)
                for i in range(signature_):                    
                    assign(b, fi.sub(i))
                    sol.solve(A, f0.vector(), b.vector())
                    assign(Udesti.sub(i), f0)
                    
        else:
            if type(signature_) is list:
                for i in range(len(signature_)):
                    fj = fi.sub(i)
                    Vj = Vi.sub(i)
                    Udestj = Udesti.sub(i)
                    recproj(fj, Vj, Udestj)

    recproj(L, Fdest, Udest)
    return Udest
