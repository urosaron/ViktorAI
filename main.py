#!/usr/bin/env python3
"""
ViktorAI - A chatbot that embodies Viktor from Arcane Season 1

This is the main entry point for the ViktorAI chatbot. It loads the character data,
initializes the chatbot, and provides a command-line interface for interacting with Viktor.
"""

import argparse
import sys
from src.viktor_ai import ViktorAI
from src.config import Config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ViktorAI - Viktor from Arcane Season 1"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="The Ollama model to use (default: llama3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (default: 0.7)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum tokens for response generation (default: 500)",
    )

    # Response classifier arguments
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        help="Enable response quality classification with PyTorch (default: False)",
    )
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="Disable response quality classification even if available",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum acceptable score for responses (default: 0.6)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for response classification",
    )

    return parser.parse_args()


def main():
    """Main function to run the ViktorAI chatbot."""
    args = parse_arguments()

    # Initialize configuration
    config = Config(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        # Response classifier settings
        use_response_classifier=args.use_classifier and not args.no_classifier,
        min_response_score=args.min_score,
        debug=args.debug,
    )

    # Initialize the chatbot
    print("Initializing ViktorAI...")
    viktor_ai = ViktorAI(config)

    # Welcome message
    print("\n" + "=" * 50)
    print(f"Welcome to ViktorAI - Viktor from Arcane Season 1")
    print(f"Using model: {config.model_name}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")

    # Show classifier info if enabled
    if config.use_response_classifier:
        print(
            f"Response quality classifier: Enabled (min score: {config.min_response_score})"
        )
    else:
        print("Response quality classifier: Disabled")

    print("Type 'exit', 'quit', or Ctrl+C to end the conversation.")
    print("=" * 50 + "\n")

    # Main conversation loop
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("\nViktor: Farewell. The work continues.")
                break

            # Get response from Viktor
            response = viktor_ai.generate_response(user_input)

            # Print Viktor's response
            print(f"\nViktor: {response}")

    except KeyboardInterrupt:
        print("\n\nConversation terminated. The glorious evolution awaits.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
