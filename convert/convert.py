import pandas as pd
import os

def convert_csv(file_name):
    # before 폴더의 CSV 파일 불러오기
    before_path = os.path.join("convert/before", file_name)
    df = pd.read_csv(before_path)

    df.columns = ['ID', 'click']
    
    # click 열을 0 또는 1로 변환
    df['click'] = df['click'].apply(lambda x: 1 if x >= 0.5 else 0)
    
    # 변환된 CSV 파일을 after 폴더에 저장
    after_path = os.path.join("convert/after", file_name)
    df.to_csv(after_path, index=False)
    
    # 0과 1의 개수, 1의 비율 출력
    click_counts = df['click'].value_counts()
    count_0 = click_counts.get(0, 0)
    count_1 = click_counts.get(1, 0)
    ratio_1 = count_1 / len(df) if len(df) > 0 else 0
    
    print(f"변환 완료: {file_name}")
    print(f"0의 개수: {count_0}, 1의 개수: {count_1}, 1의 비율: {ratio_1:.2%}")


# before 폴더에 있는 파일들을 하나씩 변환
for file in os.listdir("convert/before"):
    if file.endswith(".csv"):
        convert_csv(file)


# python convert/convert.py
