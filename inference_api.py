"""
IRIS classification - command line inference via API
"""

import sys
import json
import argparse
import requests


# Default examples
# api_url = "http://localhost:8080/2015-03-31/functions/function/invocations"


def arg_parser():
    """Parse arguments"""

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="IRIS classification inference via API call")
    # Add arguments
    parser.add_argument(
        "-u", "--url", type=str, help="URL to the server (with endpoint location)", required=True
    )
    parser.add_argument("-d", "--data", type=str, help="Input data", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    return parser


def main(args=None):
    """Main function"""

    args = arg_parser().parse_args(args)
    # Use the arguments
    if args.verbose:
        print(f"Input data: {args.data}")
        print(f"Input data type: {type(args.data)}")

    # Send request to API
    response = requests.post(args.url, json=json.loads(args.data), timeout=60)

    if response.status_code == 200:
        # Process the response
        processed_data = json.loads(response.content)
        print("processed_data", processed_data)
    else:
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
