import os
import argparse

# load env with dotenv
from dotenv import load_dotenv
load_dotenv()

assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

from src.user_io.user_io_sm import UserIoSm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default path is "/home/edward/ste001/domo_datasets/5284a372-c323-4d77-af30-d9c4d1b2624c.csv"
    parser.add_argument("--csv_path", type=str, default="/home/edward/ste001/feature_df.csv")
    args = parser.parse_args()
    user_io_sm = UserIoSm(csv_path=args.csv_path)
    user_io_sm.run()

