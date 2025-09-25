"""Entry point for launching the AFM corrosion pit tracker GUI."""

from afm_tracker import SmartPitTracker


def main():
    """Start the interactive tracker and keep the CLI output tidy."""
    tracker = SmartPitTracker()
    if tracker.images:
        tracker.run_interactive_gui()
        print("\nAnalysis complete!")
    else:
        print("No images loaded. Exiting.")


if __name__ == "__main__":
    main()
