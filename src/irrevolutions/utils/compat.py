def initial_mode_from_spectrum(spectrum):
    """Extract the first initial mode vector from older spectrum payloads."""
    if not spectrum:
        return None

    first_mode = spectrum[0]
    if isinstance(first_mode, dict):
        return first_mode.get("xk")
    return first_mode
