import io
import timeit

from lib.flc import FLC


def bench(file: io.BufferedIOBase) -> None:
    file.seek(0, io.SEEK_SET)
    FLC(file)


if __name__ == "__main__":
    with open("extract/JUKEBOX.SI_64.flc", "rb") as file:
        execution_time = timeit.timeit(lambda: bench(file), number=20)
        print(f"Execution time: {execution_time:.2f} seconds")
