import time
import subprocess
from session.create import create_session
from load.main import get_dataset, adjust_string_columns, adjust_num_columns
from experiment.main import experimental_case
from util.logger import trace_log
from util.env_vars import config

if __name__ == '__main__':
    timestamp = str(time.time())

    try:
        spark_session = create_session()

        dataset = get_dataset(
            spark_session,
            "final_animedataset.csv", ",",
            columns_to_drop=["username", "anime_id", "my_score", "user_id", "title"])

        dataset = adjust_num_columns(
            dataset, ["score", "scored_by", "rank", "popularity"])
        dataset, log = adjust_string_columns(
            dataset, ["genre", "gender", "type", "source"])
        trace_log(timestamp, log)

        dataset = dataset.dropna()

        experimental_case(dataset, timestamp)

        spark_session.stop()

        subprocess.run(
            ['notify-send',
             'Process ended',
             'General process has ended, check results.'])

    except Exception as e:
        print("An error interrupeted the script execution")
        print(e)
        subprocess.run(
            ['notify-send',
             "An error interrupeted the script execution",
             f'{e}'])
