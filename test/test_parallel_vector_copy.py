from petsc4py import PETSc

# Initialize PETSc
PETSc.Sys.popErrorHandler()

# Create a parallel vector
comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()

# Create a vector on each process
x_local = PETSc.Vec().create(comm=comm)
x_local.setSizes(5)
x_local.setFromOptions()
print([rank * 5 + i for i in range(5)])
x_local.setValues(range(5), [rank * 5 + i for i in range(5)], addv=False)
x_local.assemblyBegin()
x_local.assemblyEnd()

# Create a global vector
x_global = PETSc.Vec().create(comm=comm)
x_global.setSizes(size * 5)
x_global.setFromOptions()

# Perform parallel vector copy
x_global.copy(x_local)

# Print the global vector on each process
print(f"Process {rank}: {x_global.getArray()}")

# Cleanup
x_local.destroy()
x_global.destroy()
