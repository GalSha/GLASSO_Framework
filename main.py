from utils.args import get_args
from tasks.common import run_task
import time
from datetime import timedelta


def main():
    # parse configuration
    args = get_args()

    # start timer
    start = time.time()

    # start of task
    run_task(args)
    # end of task

    # end timer
    end = time.time()
    elapsed = end - start
    print("Elapsed time = " + str(timedelta(seconds=elapsed)))


if __name__ == "__main__":
    main()
