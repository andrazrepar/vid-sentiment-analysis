# -*- coding: utf-8 -*-
import argparse
from transformers import pipeline


def analyze(model, in_file, out_file ):
    data = []
    with open(in_file, "r") as f:
        for line in f:
            data.append(line.strip())
    results = model(data)

    with open(out_file, "w") as wf:
        for i in range(len(results)):
            wf.write(f'{data[i]}\t{results[i]["label"]}\t{results[i]["score"]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment analysis with HuggingFace')
    parser.add_argument('--input', help='input file')
    parser.add_argument('--output', help='output file')
    args = parser.parse_args()
    print(args)

    sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    analyze(sentiment_pipeline, args.input, args.output)

