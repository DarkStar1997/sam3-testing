import argparse
import json
import os

from PIL import Image

from locateanything_worker import LocateAnythingWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="LocateAnything-3B single-image inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True,
                        help="comma-separated categories (detect) or free-text phrase (other tasks)")
    parser.add_argument("--task", default="detect",
                        choices=["detect", "ground", "point", "text", "gui"])
    parser.add_argument("--gui-type", default="box", choices=["box", "point"])
    parser.add_argument("--mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--model", default=os.environ.get("MODEL_DIR", "/opt/LocateAnything-3B"))
    parser.add_argument("--json", action="store_true", help="print parsed boxes/points as JSON")
    args = parser.parse_args()

    worker = LocateAnythingWorker(args.model)
    image = Image.open(args.image).convert("RGB")
    kwargs = {"generation_mode": args.mode, "max_new_tokens": args.max_new_tokens}

    if args.task == "detect":
        result = worker.detect(image, [c.strip() for c in args.query.split(",")], **kwargs)
    elif args.task == "ground":
        result = worker.ground_multi(image, args.query, **kwargs)
    elif args.task == "point":
        result = worker.point(image, args.query, **kwargs)
    elif args.task == "text":
        result = worker.detect_text(image, **kwargs)
    else:
        result = worker.ground_gui(image, args.query, output_type=args.gui_type, **kwargs)

    print(result["answer"])
    if args.json:
        parsed = (worker.parse_points if args.task in ("point", "gui") and args.gui_type == "point"
                  else worker.parse_boxes)
        print(json.dumps(parsed(result["answer"], image.width, image.height), indent=2))


if __name__ == "__main__":
    main()
