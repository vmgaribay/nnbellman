import torch
import model.model as module_arch
import tensorflow as tf
import os
import re
from datetime import datetime

model_path = "/Users/victoria/Documents/Scripts/Python/NNFunction/Trial3/saved/i_a/S21/models/i_a_FiveLayer_2048/0402_173814/model_best.pth"


#modelinfo = torch.load(model_path,map_location=torch.device('mps'))

#print(modelinfo)

#config = modelinfo['config']

#model = getattr(module_arch,config['arch']['type'])(**config['arch']['args'])

#model.load_state_dict(modelinfo['state_dict'])
from tensorflow.core.util import event_pb2
from google.protobuf import text_format

def print_event_file(event_file_path):
    with open(event_file_path, "rb") as f:
        for event in f:
            event_data = event_pb2.Event()
            event_data.ParseFromString(event)
            print(text_format.MessageToString(event_data))

# Replace "event_file_path" with the path to your event file
print_event_file("/Users/victoria/Documents/Scripts/Python/NNFunction/Trial3/saved/consumption/S21/log/TestModel/0409_171030/events.out.tfevents.1712675434.Victorias-MacBook-Pro.local.95453.0")
'''
# Define the log file path
log_file = "/Users/victoria/Documents/Scripts/Python/NNFunction/Trial3/saved/i_a/S21/log/i_a_FiveLayer_512/0402_173406/info.log"

# Define the event file path
event_file = "/Users/victoria/Documents/Scripts/Python/NNFunction/Trial3/saved/i_a/S21/log/i_a_FiveLayer_512/0402_173406/"


# Define the regex patterns for extracting relevant information
patterns = {
    "epoch": r"epoch\s+:\s+(\d+)",
    "loss": r"loss\s+:\s+([\d.]+)",
    "category_accuracy": r"category_accuracy\s+:\s+([\d.]+)",
    "no_adapt_falsepositive": r"no_adapt_falsepositive\s+:\s+(\d+)",
    "no_adapt_falsenegative": r"no_adapt_falsenegative\s+:\s+(\d+)",
    "low_adapt_falsepositive": r"low_adapt_falsepositive\s+:\s+(\d+)",
    "low_adapt_falsenegative": r"low_adapt_falsenegative\s+:\s+(\d+)",
    "high_adapt_falsepositive": r"high_adapt_falsepositive\s+:\s+(\d+)",
    "high_adapt_falsenegative": r"high_adapt_falsenegative\s+:\s+(\d+)",
    "val_loss": r"val_loss\s+:\s+([\d.]+)",
    "val_category_accuracy": r"val_category_accuracy\s+:\s+([\d.]+)",
    "val_no_adapt_falsepositive": r"val_no_adapt_falsepositive\s+:\s+(\d+)",
    "val_no_adapt_falsenegative": r"val_no_adapt_falsenegative\s+:\s+(\d+)",
    "val_low_adapt_falsepositive": r"val_low_adapt_falsepositive\s+:\s+(\d+)",
    "val_low_adapt_falsenegative": r"val_low_adapt_falsenegative\s+:\s+(\d+)",
    "val_high_adapt_falsepositive": r"val_high_adapt_falsepositive\s+:\s+(\d+)",
    "val_high_adapt_falsenegative": r"val_high_adapt_falsenegative\s+:\s+(\d+)"
}

# Function to extract relevant information from log lines
def extract_info(line):
    info = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            info[key] = match.group(1)
    return info

# Create TensorFlow SummaryWriter
writer = tf.summary.create_file_writer(event_file)

# Parse log file and write to event file
with open(log_file, "r") as file:
    for line in file:
        line = line.strip()
        if line.startswith("20"):
            timestamp = datetime.strptime(line.split(',')[0], "%Y-%m-%d %H:%M:%S")
            step = int((timestamp - datetime(1970, 1, 1)).total_seconds())
            info = extract_info(line)
            with writer.as_default():
                for key, value in info.items():
                    tf.summary.scalar(key, float(value), step=step)
writer.close()
'''
""" 
def find_log_files(root_folder):
    log_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))
    return log_files

# Convert extracted data to TensorFlow Summary events
def create_summary_events(data_dict,path):
    writer = tf.summary.create_file_writer(path)  # Create a summary writer
    with writer.as_default():
        for metric_name, metric_values in data_dict.items():
            # Create a summary for each metric
            for step, value in enumerate(metric_values):
                tf.summary.scalar(metric_name, value, step=step)
    writer.flush()

# Main function
def main():
    root_folder = "/Users/victoria/Documents/Scripts/Python/NNFunction/Trial3/saved/i_a/S21/"
    root_folder = "/Users/victoria/Documents/Scripts/Python/NNFunction/Trial3/saved/i_a/S21/log/i_a_FiveLayer_512/0402_173406/"
    log_files = find_log_files(root_folder)
    for log_file in log_files:
        data_dict = parse_log_file(log_file)
        print (log_file)
        print (data_dict)
        create_summary_events(data_dict, os.path.dirname(log_file))
    

if __name__ == "__main__":
    print(tf.__version__)

    main() """