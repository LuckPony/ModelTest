
def test():
    print("Hello World")

def main():
    test_noisy_file = "data/2_percent_noise/sub-103818__dwi_2%noise.nii.gz"
    test_clean_file = "data/gt/sub-103818__dwi.nii.gz"
    model_path = "model/"
    test()

if __name__ == "__main__":
    main()