UNIT_LENGTH = {
    "angstrom": 1,
    "bohr": 1.8897259886,
    "nm": 0.1
}
UNIT_ENERGY = {
    "ev": 1,
    "hartree": 0.0367493
}


def get_conversion_factor(unit, from_unit, to_unit):
    if unit.lower() == "length":
        conversion_table = UNIT_LENGTH
    elif unit.lower() == "energy":
        conversion_table = UNIT_ENERGY
    return conversion_table.get(to_unit) / conversion_table.get(from_unit)
