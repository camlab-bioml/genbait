def calculate_subset_range(n_baits_to_select, delta=10):
    """
    Calculate the valid range of subset sizes for bait selection.

    This function defines a range of subset sizes based on the number of desired baits to select.
    It provides a range from (n_baits_to_select - delta) to n_baits_to_select.

    Args:
    - n_baits_to_select (int): The target number of baits to be selected.
    - delta (int, optional): The allowable range around the target number of baits to be selected.
      Default is 10.

    Returns:
    - tuple: A tuple containing the minimum and maximum subset size (min_size, max_size).
    """
    # Returns a tuple of the subset range, i.e., (n_baits_to_select - delta, n_baits_to_select).
    # delta specifies the buffer on how much smaller the subset size can be.
    return (n_baits_to_select - delta, n_baits_to_select)

