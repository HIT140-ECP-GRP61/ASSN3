from feature_engineering_BAI import build_bai

if __name__ == "__main__":
    build_bai(
        d1_path="dataset1.csv",
        d2_path="dataset2.csv",
        out_path="engineered_bai.csv",
        k=1.0,
        do_winsor=True
    )