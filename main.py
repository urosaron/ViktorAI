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
    
    # Model arguments
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="The Ollama model to use (default: llama3)",
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (default: 0.7)",
    )
    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens for response generation (default: 500)",
    )

    # Brain integration arguments
    brain_group = parser.add_argument_group("ViktorBrain Integration")
    brain_group.add_argument(
        "--brain-url",
        type=str,
        default="http://localhost:8000",
        help="ViktorBrain API URL (default: http://localhost:8000)",
    )
    brain_group.add_argument(
        "--brain-neurons",
        type=int,
        default=1000,
        help="Number of neurons for brain simulation (default: 1000)",
    )
    brain_group.add_argument(
        "--brain-density",
        type=float,
        default=0.1,
        help="Connection density for brain simulation (default: 0.1)",
    )
    brain_group.add_argument(
        "--brain-activity",
        type=float,
        default=0.02,
        help="Spontaneous activity for brain simulation (default: 0.02)",
    )
    brain_group.add_argument(
        "--no-brain",
        action="store_true",
        help="Disable ViktorBrain integration",
    )

    # Response classifier arguments
    classifier_group = parser.add_argument_group("Response Classifier")
    classifier_group.add_argument(
        "--use-classifier",
        action="store_true",
        help="Enable response quality classification with PyTorch (default: False)",
    )
    classifier_group.add_argument(
        "--no-classifier",
        action="store_true",
        help="Disable response quality classification even if available",
    )
    classifier_group.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum acceptable score for responses (default: 0.6)",
    )
    classifier_group.add_argument(
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
        # Model settings
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        
        # Response classifier settings
        use_response_classifier=args.use_classifier and not args.no_classifier,
        min_response_score=args.min_score,
        debug=args.debug,
        
        # Brain integration settings
        brain_api_url=args.brain_url,
        brain_neurons=args.brain_neurons,
        brain_connection_density=args.brain_density,
        brain_spontaneous_activity=args.brain_activity,
        use_brain=not args.no_brain,
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

    # Show brain integration info
    if config.use_brain:
        print(f"ViktorBrain integration: Enabled")
        print(f"  - API URL: {config.brain_api_url}")
        print(f"  - Neurons: {config.brain_neurons}")
    else:
        print("ViktorBrain integration: Disabled")

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
