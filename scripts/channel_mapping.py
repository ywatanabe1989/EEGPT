import mngs

CONFIG = mngs.io.load("./config/global.yaml")

def get_unique_channels():
    LPATHS_CHANNELS = mngs.io.find(CONFIG["DATA_DIR"], "*channels*.tsv")
    chs_all = []
    for lpath_tsv in LPATHS_CHANNELS:
        try:
            chs = np.array(mngs.io.load(lpath_tsv).name)
        except Exception as e:
            print(f"\n{lpath_tsv}")
            print(e)
            chs = "Unknown"
        chs_all.append(chs)
    # Gets unique channel patterns by converting them to string
    chs_all = np.unique(
        [mngs.gen.connect_strs(natsorted(chs), filler="-") for chs in chs_all]
    )
    # Restore the strings to numpy expressions
    chs_all = [np.array(chs.split("-")) for chs in chs_all]
    # To upper cases
    chs_all = [np.char.upper(chs) for chs in chs_all]
    return chs_all


# def print_missing_and_extra_channels():
#     for dataset_channels in chs_uq:
#         # Identify missing channels
#         missing_channels = set(CONFIG["TGT_MONTAGE"]) - set(dataset_channels)
#         print(f"Missing channels: {missing_channels}")

#         # Identify extra channels
#         print()
#         extra_channels = set(dataset_channels) - set(CONFIG["TGT_MONTAGE"])
#         print(f"Extra channels: {extra_channels}")
#         print()


def map_channels(dataset_channels, channel_mapping):
    # Map dataset channels to standardized channel names
    standardized_channels = [
        channel_mapping.get(ch, ch) for ch in dataset_channels
    ]
    return standardized_channels


def print_missing_and_extra_channels(chs_uq, channel_mapping):
    for dataset_channels in chs_uq:
        standardized_channels = map_channels(dataset_channels, channel_mapping)

        # Identify missing channels
        missing_channels = set(CONFIG["TGT_MONTAGE"]) - set(
            standardized_channels
        )
        print(f"Missing channels: {missing_channels}")

        # Identify extra channels
        extra_channels = set(standardized_channels) - set(
            CONFIG["TGT_MONTAGE"]
        )
        print(f"Extra channels: {extra_channels}")
        print()


if __name__ == "__main__":

    from natsort import natsorted

    chs_uq = get_unique_channels()

    # print(CONFIG["TGT_MONTAGE"])
    # [
    #     "FP1",
    #     "F3",
    #     "C3",
    #     "P3",
    #     "O1",
    #     "FP2",
    #     "F4",
    #     "C4",
    #     "P4",
    #     "O2",
    #     "F7",
    #     "T7",
    #     "P7",
    #     "F8",
    #     "T8",
    #     "P8",
    #     "Fz",
    #     "Cz",
    #     "Pz",
    #     "A1",
    #     "A2",
    # ]

    # Example channel mapping (you would need to define this based on your datasets)
    channel_mapping = {
        "T3": "T7",
        "T5": "P7",
        "T4": "T8",
        "T6": "P8",
        # Add other mappings as needed
    }

    # Example usage
    print_missing_and_extra_channels(chs_uq, channel_mapping)

    # Overall, TGT_MONTAGE and TGT_TRANVERSE_BIPOLAR_MONTAGE are achieved by the Channel mapping
