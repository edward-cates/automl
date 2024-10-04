import os
import argparse

# load env with dotenv
from dotenv import load_dotenv
load_dotenv()

assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

from src.user_io.user_io_sm import UserIoSm

if __name__ == "__main__":
    user_io_sm = UserIoSm()
    while True:
        user_io_sm.step()
