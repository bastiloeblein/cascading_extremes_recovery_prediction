def print_channel_info(s2, s1, era5, stat):
    print("\n" + " channel mapping ".center(40, "="))
    current_idx = 0

    # Sentinel-2
    for v in s2:
        print(f"CH {current_idx:02d}: S2_{v}")
        current_idx += 1
    # Sentinel-1
    for v in s1:
        print(f"CH {current_idx:02d}: S1_{v}")
        current_idx += 1
    # ERA5
    for v in era5:
        print(f"CH {current_idx:02d}: ERA5_{v}")
        current_idx += 1
    # Masks
    print(f"CH {current_idx:02d}: Mask_S2")
    print(f"CH {current_idx+1:02d}: Mask_S1")
    current_idx += 2
    # Static
    print(f"CH {current_idx:02d} - {current_idx+11:02d}: ESA_Landcover (One-Hot)")
    current_idx += 12
    for v in stat:
        if v != "ESA_LC":  # ESA_LC ist schon drin
            print(f"CH {current_idx:02d}: Static_{v}")
            current_idx += 1
    print("=" * 40 + "\n")
