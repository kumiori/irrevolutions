import logging


def setup_logger_mpi(root_priority: int = logging.INFO):
    from mpi4py import MPI
    import dolfinx
    class MPIFormatter(logging.Formatter):
        def format(self, record):
            record.rank = MPI.COMM_WORLD.Get_rank()
            record.size = MPI.COMM_WORLD.Get_size()
            return super(MPIFormatter, self).format(record)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Get_size()

    # Desired log level for the root process (rank 0)
    root_process_log_level = logging.INFO  # Adjust as needed

    logger = logging.getLogger('E•volver')
    logger.setLevel(root_process_log_level if rank == 0 else logging.WARNING)

    # StreamHandler to log messages to the console
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('evolution.log')

    # formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    formatter = MPIFormatter('%(asctime)s  [Rank %(rank)d, Size %(size)d]  - %(name)s - [%(levelname)s] - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # file_handler.setLevel(logging.INFO)
    file_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)
    console_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log messages, and only the root process will log.
    logger.info("The root process spawning an evolution computation (rank 0)")
    logger.info(
    f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

    logger.critical("Critical message")
    if rank == 1:
        logger.error(f"{rank} Error message")
        logger.info(f"{rank} Info message")
        
    # logger.warning("Warning message")
    
    return logger

if __name__ == "__main__":
    setup_logger_mpi()