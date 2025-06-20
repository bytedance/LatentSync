import argparse
import os
from latentsync.pipelines.alignment_pipeline import AlignmentPipeline


def main(args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")

    print(f"Input video path: {args.video_path}")

    pipeline = AlignmentPipeline()

    pipeline(
        video_path=args.video_path,
        width=args.dim,
        height=args.dim,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--dim", type=int, required=True)
    args = parser.parse_args()
    main(args)
