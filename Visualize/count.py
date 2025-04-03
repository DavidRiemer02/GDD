import chardet



with open("TestData/realData/meets.csv", 'rb') as f:
    result = chardet.detect(f.read(10000))
    print(result['encoding'])