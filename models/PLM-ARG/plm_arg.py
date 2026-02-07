# -*- coding: utf-8 -*-
"""
Created on Sun Jan  29 19:54:55 2023

@author: Jun Wu
"""
import argparse
from train import train
from predict import predict

def run_train(args):
    train(in_fasta=args.input_file, batch_size=args.batch_size, maxlen=200, min_seq=args.min_seq,
          arg_model = args.arg_model, cat_model = args.cat_model, cat_index= args.cat_index)

def run_predict(args):
    predict(in_fasta=args.input_file, batch_size= args.batch_size, maxlen=200, min_prob=args.min_prob,
            arg_model = args.arg_model, cat_model = args.cat_model, cat_index= args.cat_index,
            output_file = args.output_file)

def main():
    parser = argparse.ArgumentParser(prog='PLM-ARG')
    subparsers = parser.add_subparsers()
    # run the prediction section
    plm_arg_p = subparsers.add_parser("predict", help="Predict ARG from genes or ORF")
    plm_arg_p.add_argument('-i', '--input-file', required=True, help='Input file (Fasta input file)')
    plm_arg_p.add_argument('--arg-model', default='models/arg_model.pkl', help='Model for ARG identification')
    plm_arg_p.add_argument('--cat-model', default='models/cat_model.pkl', help='Model for resistance category classification')
    plm_arg_p.add_argument('--cat-index', default='models/Category_Index.csv', help='File for the resistance category index')
    plm_arg_p.add_argument('-o', '--output-file', required=True, help='Output file where to store results')
    plm_arg_p.add_argument('--min-prob', default=0.5, type=float, help='Minimum probability cutoff [Default: 0.5]')
    plm_arg_p.add_argument('-b','--batch-size', default=10, type=float,help='Number of the samples fed to the model iteratively [Default: 10]')
    plm_arg_p.set_defaults(func=run_predict)

    # run the train section
    plm_arg_t = subparsers.add_parser("train", help="Retrain the PLM-ARG models")
    plm_arg_t.add_argument('-i', '--input-file', required=True, help='Input file (Fasta input file)')
    plm_arg_t.add_argument('--arg-model', default='models/arg_model.pkl', help='Model for ARG identification')
    plm_arg_t.add_argument('--cat-model', default='models/cat_model.pkl',
                         help='Model for resistance category classification')
    plm_arg_t.add_argument('--cat-index', default='models/Category_Index.csv',
                         help='File for the resistance category index')
    plm_arg_t.add_argument('--min-seq', default=50, type=float, help='Minimal sequence number for the category training [Default: 50]')
    plm_arg_t.add_argument('-b', '--batch-size', default=10, type=float,
                         help='Number of the samples fed to the model iteratively [Default: 10]')
    plm_arg_t.set_defaults(func=run_train)

    # Get all arguments
    args = parser.parse_args()
    args.func(args)
    pass

if __name__ == '__main__':
    main()
