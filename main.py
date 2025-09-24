from afm_tracker import SmartPitTracker


def main():
    tracker = SmartPitTracker()
    if tracker.images:
        tracker.run_interactive_gui()
        print("\nAnalysis complete!")
    else:
        print("No images loaded. Exiting.")


if __name__ == "__main__":
    main()
