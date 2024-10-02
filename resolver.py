import pandas as pd
import sys
import os
import json
import re

script_dir = os.path.dirname(os.path.realpath(__file__))
item_fname = os.path.join(script_dir, 'data/movie_final.csv')

def read_csv():
  movies_df=pd.read_csv(item_fname, encoding='utf-8')
  movies_df = movies_df.fillna("") # NaN을 공백으로 채워줌
  return movies_df[1:]

def random_items(count):
  movies_df=read_csv()
  result_items = movies_df.sample(n=count).to_dict("records")
  return json.dumps(result_items)

def latest_items(count):
  # CSV 파일을 읽어옴
  movies_df = read_csv()
  # 영화 제목에서 연도를 추출하는 함수 (예: "Journey to the Center of the Earth (1959)" -> 1959)
  def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return None  # 연도가 없으면 None 반환
  # 연도 정보를 추출해서 새로운 'year' 컬럼에 추가
  movies_df['year'] = movies_df['title'].apply(extract_year)
  # 연도 기준으로 내림차순으로 정렬하고 최신 10개 항목 선택
  latest_movies_df = movies_df.sort_values(by='year', ascending=False).head(count)
  # 결과를 딕셔너리 형태로 반환
  result_items = latest_movies_df.to_dict("records")
  return json.dumps(result_items)

def genres_items(genre, count):
  movies_df = read_csv()
  # genres가 "Fantasy"인 것만 필터링
  # na=False는 결측값(NaN)을 False로 처리하므로, 결측값은 필터링되지 않도록 합니다.
  # 주어진 genre가 장르 목록에 포함되어 있는지 확인하기 위해 정규 표현식 사용
  # ^와 $를 사용하여 전체 문자열로 일치하도록 하고, |를 사용하여 다양한 장르를 구분
  # 장르가 |로 구분된 여러 항목 중 하나로 존재하는지 체크
  # (?<=\|){genre}(?=\|): 주어진 장르가 중간에 있을 때
  # ^{genre}(?=\|): 문자열의 시작에 장르가 있을 때
  # {genre}(?=$): 문자열의 끝에 장르가 있을 때
  # ^{genre}$: 문자열이 정확히 주어진 장르와 같을 때
  filtered_df = movies_df[movies_df['genres'].str.contains(f'(?<=\|){genre}(?=\|)|^{genre}(?=\|)|{genre}(?=$)|^{genre}$', na=False)]

  # filtered_df가 비어 있는지 확인
  if filtered_df.empty:
    error_message = {"error": f"No movies found for genre: {genre}"}
    return json.dumps(error_message)  # JSON 형식으로 변환하여 반환

  # 필터링된 데이터 중에서 샘플을 뽑아옴
  # count가 filtered_df의 길이보다 크면 그 길이만큼 샘플링
  sample_count = min(count, len(filtered_df))
  result_items = filtered_df.sample(n=sample_count).to_dict("records")
  return json.dumps(result_items)

if __name__ == "__main__":
  try:
    match sys.argv[1]:
      case "random":
        print(random_items(int(sys.argv[2])))
      case "latest":
        print(latest_items(int(sys.argv[2])))
      case "genres":
        print(genres_items(sys.argv[2], int(sys.argv[3])))
      case _:
        print(json.dumps({"error": "Invalid command provided"}))
  except ValueError:
    print(json.dumps({"error": "Invalid arguments"}))