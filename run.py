import argparse

from src.user_io.user_io_sm import UserIoSm

if __name__ == "__main__":
    user_io_sm = UserIoSm()
    while True:
        user_io_sm.step()
