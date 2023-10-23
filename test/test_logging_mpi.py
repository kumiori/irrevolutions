import logging
from mpi4py import MPI


def setup_logger_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set the desired log level for the root process (rank 0)
    root_process_log_level = logging.INFO  # Adjust as needed

    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)

    # Configure the logger (e.g., add handlers, formatters)
    # ...

    # Add a StreamHandler to log messages to the console (or you can use other handlers)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(root_process_log_level if rank == 0 else logging.CRITICAL)
    logger.addHandler(console_handler)

    # Set the formatter for the handler
    # ...

    if rank == 0:
    # Configure additional handlers or custom settings for the root process
    # ...
        print(rank)

    # Now, you can use the 'logger' object to log messages, and only the root process will log.
    logger.info("This message will be logged only on the root process (rank 0)")
    logger.info(f"This is process {rank} reporting")

    return logger


if __name__ == "__main__":
    setup_logger_mpi()