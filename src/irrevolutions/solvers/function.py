import typing

import dolfinx
import numpy
import ufl
from dolfinx.fem import Function
from petsc4py import PETSc


def extract_blocks(
    form,
    test_functions: typing.List[ufl.Argument],
    trial_functions: typing.List[ufl.Argument] = None,
):
    """Extract blocks from a monolithic UFL form.

    Parameters
    ----------
    form
    test_functions
    trial_functions: optional

    Returns
    -------
    Splitted UFL form in the order determined by the passed test and trial functions.
    If no `trial_functions` are provided returns a list, otherwise returns list of lists.

    """
    # Check for distinct test functions
    if len(test_functions) != len(set(test_functions)):
        raise RuntimeError(
            "Duplicate test functions detected. Create TestFunction(s) from separate FunctionSpace(s)!"
        )

    # Prepare empty block matrices list
    if trial_functions is not None:
        blocks = [
            [None for i in range(len(test_functions))]
            for j in range(len(trial_functions))
        ]
    else:
        blocks = [None for i in range(len(test_functions))]

    for i, tef in enumerate(test_functions):
        if trial_functions is not None:
            for j, trf in enumerate(trial_functions):
                to_null = dict()

                # Dictionary mapping the other trial functions
                # to zero
                for item in trial_functions:
                    if item != trf:
                        to_null[item] = ufl.zero(item.ufl_shape)

                # Dictionary mapping the other test functions
                # to zero
                for item in test_functions:
                    if item != tef:
                        to_null[item] = ufl.zero(item.ufl_shape)

                blocks[i][j] = ufl.replace(form, to_null)
        else:
            to_null = dict()

            # Dictionary mapping the other test functions
            # to zero
            for item in test_functions:
                if item != tef:
                    to_null[item] = ufl.zero(item.ufl_shape)

            blocks[i] = ufl.replace(form, to_null)

    return blocks


def functions_to_vec(u: typing.List[Function], x):
    """Copies functions into block vector."""
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            u[i].x.petsc_vec.copy(subvec)
            subvec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
    else:
        offset = 0
        for i in range(len(u)):
            size_local = u[i].x.petsc_vec.getLocalSize()
            with x.localForm() as loc:
                loc.array[offset : offset + size_local] = u[i].x.petsc_vec.array_r
            offset += size_local
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def vec_to_functions(x, u: typing.List[dolfinx.fem.Function]):
    """Copies block vector into functions."""

    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            subvec.copy(u[i].x.petsc_vec)
            u[i].x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
    else:
        offset = 0
        for i in range(len(u)):
            size_local = u[i].x.petsc_vec.getLocalSize()
            u[i].x.petsc_vec.array[:] = x.array_r[offset : offset + size_local]
            offset += size_local
            u[i].x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )


def unroll_dofs(dofs, block_size):
    """Unroll blocked dofs."""
    arr = block_size * numpy.repeat(dofs, block_size).reshape(
        -1, block_size
    ) + numpy.arange(block_size)
    return arr.flatten().astype(dofs.dtype)
