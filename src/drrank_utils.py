######################
# Utilities functions#
######################

def isfloat(num):
    """
    Check whether a number is float
    """
    try:
        float(num)
        return True
    except ValueError:
        return False
