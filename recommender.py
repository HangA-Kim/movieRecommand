import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix # COO 형식의 행렬을 만들기 위해 사용
from implicit.als import AlternatingLeastSquares # 알고리즘 사용을 위해 import
import pickle # 모델 저장을 위해 import
import sys
import json
import threadpoolctl
from scipy.sparse import coo_matrix, csr_matrix 
# coo_matrix : COO 형식의 행렬을 만들기 위해 사용, csr_matrix : CSR 형식의 행렬을 만들기 위해 사용


threadpoolctl.threadpool_limits(1, "blas") # Python에서 실행되는 과학 계산 라이브러리들이 사용하는 다중 스레드(multithreading)를 제어하기 위한 함수 호출
# threadpool_limits(1, "blas")는 blas 라이브러리가 사용할 수 있는 스레드 수를 1로 제한한다는 의미입니다. 이렇게 하면 병렬 처리 대신 단일 스레드로 작업을 처리하게 됩니다.
# 병렬화된 연산이 오히려 성능을 저하시킬 수 있는 상황에서, 다중 스레드를 제한하여 더 나은 성능을 기대할 수 있습니다.


script_dir = os.path.dirname(os.path.realpath(__file__))
saved_model_fname = os.path.join(script_dir, 'model/finalized_model.sav')
ratings_fname = os.path.join(script_dir, 'data/ratings.csv')
final_fname = os.path.join(script_dir, 'data/movie_final.csv')
weight = 10 # 가중치 값 설정 (10으로 설정) : 사용자의 평점을 가중치로 사용하여 모델을 학습


########################################################################################
# 데이터를 학습하여 모델을 생성하는 함수
# userId 와 movieId 을 이용한 평점을 학습 
########################################################################################
def model_train():
  ratings_df = pd.read_csv(ratings_fname)
  # userid와 movieid를 category 형태로 변환
  # category (범주형).  메모리 사용량이 줄어듭니다. 범주형 데이터는 고유 값만 저장하고 나머지 값은 참조를 통해 관리하므로 메모리 효율성이 높습니다.
  ratings_df["userId"] = ratings_df["userId"].astype("category")
  ratings_df["movieId"] = ratings_df["movieId"].astype("category")
  
  # 어떤 유저(userId)가 어떤 영화(movieId)에 얼마의 평점(rating)을 주었는지 행렬 형태로 표현해 주는 함수
  # coo_matrix : 희소 행렬을 COO 형식으로 생성하는 함수 (데이터, (행 인덱스, 열인덱스))
  # astype(np.float32): 평점 데이터를 32비트 부동소수점(float32) 형식으로 변환
  # Create a sparse matrix of all the item/user/counts triples
  rating_matrix = coo_matrix((ratings_df["rating"].astype(np.float32), # 행렬에 들어갈 실제 값들(평점 데이터)
                              (ratings_df["movieId"].cat.codes.copy(), # 행(row) 인덱스로 사용
                              ratings_df["userId"].cat.codes.copy(),),)) # 열(column) 인덱스로 사용
  

  # AlternatingLeastSquares:  ALS 알고리즘을 구현하여 사용자와 아이템 간의 상관관계를 학습
  # **ALS(Alternating Least Squares)**는 교대 최소제곱법으로, 사용자 행렬과 아이템 행렬을 번갈아 가며 학습하는 방식

  # factors=50 : ALS 모델이 학습할 잠재 요인의 수. 사용자와 아이템을 각각 50차원의 잠재 요인 벡터로 분해한다는 의미. 숫자가 클수록 기존 데이터에 대한 정확도는 높아지지만, 과적합의 위험이 있음. 이 경우 결과는 정확하지만, 새로운 데이터에 대한 예측력은 떨어질 수 있음. 
  
  # 과적합 참조 : https://kimmaadata.tistory.com/31

  # regularization=0.01 : 정규화 항으로, 모델이 과적합하지 않도록 제어하는 역할. 값이 클수록 모델이 너무 복잡해지는 것을 방지하지만, 너무 크면 모델의 성능이 떨어질 수 있습니다.

  # iterations=50 : ALS 알고리즘을 반복 수행하는 최대 반복 횟수. 값이 클수록 정확도는 높아지지만, 시간이 오래 걸림.

  # dtype=np.float64 : 사용되는 데이터의 자료형. 기본값은 np.float64
  # float64로 설정하면 정확도가 높아지지만, 메모리 사용량이 늘어남.
  # 대규모 데이터에서 메모리 관리가 중요한 경우 float32 등으로 바꿔 성능을 개선할 수 있습니다.
  als_model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=50, dtype=np.float64)
  
  # tocsr() 메서드는 COO(Coordinate format) 형식의 행렬을 CSR(Compressed Sparse Row) 형식으로 변환합니다.
  # CSR 형식은 메모리 효율성이 뛰어나고, 행(row) 단위로 연산을 빠르게 수행할 수 있도록 최적화된 데이터 구조입니다.
  # COO와 CSR의 차이:
  # COO: 비어있지 않은 항목의 위치를 (행, 열) 쌍으로 저장. 읽기에는 효율적이지만, 계산에는 비효율적일 수 있음.
  # CSR: 행 단위로 데이터를 압축하여 저장. 행 기반 연산에 효율적이며, 빠른 수학적 연산을 가능하게 함.
  rating_matrix_csr = rating_matrix.tocsr()

  # ALS 모델 학습. 인자로는 가중치를 곱한 CSR 형식의 rating_matrix를 전달
  # 높은 가중치(weight)를 가진 평점은 모델 학습 시 더 중요한 역할을 하게 되고, 낮은 가중치는 덜 중요한 역할을 하게 됩니다.
  als_model.fit(weight * rating_matrix_csr)

  pickle.dump(als_model, open(saved_model_fname, 'wb'))
  return als_model


########################################################################################
# 검색할 movieId 를 입력하면 
# 그 영화의 평점을 높게 평가한 userId 가 다른 영화의 평점을 높게 평가한 movieId 를 추천
########################################################################################
def item_based_recommendation(movie_id):
  ratings_df = pd.read_csv(ratings_fname)
  ratings_df["userId"] = ratings_df["userId"].astype("category") 
  ratings_df["movieId"] = ratings_df["movieId"].astype("category")

  try:
    # cat.categories.get_loc() : 카테고리의 위치를 반환
    parsed_id = ratings_df["movieId"].cat.categories.get_loc(int(movie_id))
    loaded_model = pickle.load(open(saved_model_fname, 'rb')) # rb : read binary
    recs = loaded_model.similar_items(itemid=int(parsed_id), N=11) # 유사도 계산. N은 반환할 유사 아이템의 개수를 지정
    # 검색된 결과가 자기 자신도 포함되므로 자기 자신을 제외한 것을 리턴
    # 그래서 n이 11 이지만 자기 자신 빼서 총 10개가 추천됨.
    result = [int(r) for r in recs[0] if int(r) != int(movie_id)]
  except KeyError as e:
    result = []

  # 추천된 result는 movieId의 범주형 인덱스이므로, 이를 실제 movieId로 변환
  actual_movie_ids = [ratings_df["movieId"].cat.categories[r] for r in result]

  # isin(result)는 result 리스트에 포함된 영화 ID들만 선택하는 필터링 조건. 즉, result에 포함된 영화 ID가 movies_df의 movieId 열에 있는지 여부를 확인하고, 일치하는 행만 필터링.
  movies_df = pd.read_csv(final_fname)
  result_items = movies_df[movies_df["movieId"].isin(actual_movie_ids)].to_dict("records")
  return json.dumps(result_items)



########################################################################################
# 이 추천 알고리즘은 협업 필터링(Collaborative Filtering) 기반으로 동작하며, 특히 사용자 기반 협업 필터링(User-Based Collaborative Filtering) 방식을 사용.
# 주어진 설명에 따라, 1번 키(영화 ID)에 대해 4.5의 평점을 입력하면, 해당 사용자와 비슷한 평가 패턴을 가진 다른 사용자들의 정보를 기반으로 추천을 제공.

# 입력 평점 정보 처리:

# 사용자가 특정 아이템(여기서는 영화)에 대해 평점을 입력하면, 그 평점이 input_rating_dict로 전달.
# 예를 들어, input_rating_dict = {1: 4.5, 2: 3.0}는 사용자가 영화 ID 1에 대해 4.5점을, 영화 ID 2에 대해 3.0점을 줬다는 것을 의미.

# 사용자-아이템 행렬 구성:

# build_matrix_input() 함수는 입력된 평점을 기반으로 사용자-아이템 행렬을 리턴.
# 이 행렬은 희소 행렬로, 사용자가 평가한 아이템의 위치에 그 평점이 들어가고, 다른 위치는 0으로 채워짐.
# 이 과정에서 coo_matrix로 희소 행렬을 생성하여 사용자의 평가 정보를 벡터 형태로 변환.

# 유사 사용자 찾기:

# calculate_user_based() 함수는 사전에 학습된 추천 모델을 사용하여, 사용자가 입력한 평점 데이터를 기반으로 유사한 사용자들을 검색.
# 이 유사도는 사용자가 특정 영화에 대해 비슷한 평가를 한 다른 사용자들과의 유사성(코사인 유사도, 피어슨 상관계수 등)을 계산.
# user_items에는 현재 사용자의 평점 정보가 들어가고, 이 정보를 바탕으로 유사한 사용자들이 계산.

# 추천 계산:

# 유사한 사용자들을 기반으로 추천.
# 즉, 유사한 사용자들이 평점을 높게 준 영화들을 현재 사용자에게 추천.
# 이 과정에서 loaded_model.recommend() 함수가 사용되며, N개의 추천 영화를 반환.
# 이 함수는 일반적으로 협업 필터링 알고리즘을 기반으로 학습된 모델을 사용하여, 사용자가 아직 평가하지 않은 아이템들 중에서 유사한 사용자들이 선호한 아이템들을 추천.

# 결과 반환:

# 추천된 영화 목록은 아이템(영화) ID로 변환된 후, 다시 movie_df에서 해당 영화의 상세 정보를 조회하여 최종적으로 사용자에게 반환.

# 예시:
# 1번 키(영화 ID 1)에 대한 평점이 4.5인 경우 추천 방식:
# 사용자가 영화 ID 1에 대해 4.5점을 준 경우, 알고리즘은 이 영화를 선호하는 다른 사용자들과의 유사도를 계산.
# 이 영화와 비슷한 취향을 가진 다른 사용자들이 어떤 영화를 높게 평가했는지를 분석하여, 그 영화들을 추천.
# 만약 영화 ID 1을 좋아하는 다른 사용자들이 영화 ID 10과 영화 ID 15에 대해서도 높은 평점을 준 경우, 해당 영화들이 추천 목록에 포함될 가능성이 높음.
########################################################################################

def calculate_user_based(user_items, items):
  loaded_model = pickle.load(open(saved_model_fname, 'rb'))

  # userid=0 : 이 파라미터는 추천을 받을 사용자 ID. userid=0은 현재 추천을 요청하는 사용자가 "가상의 사용자"임을 나타낸다.
  # userid가 0인 경우, 사용자의 정보가 새롭게 주어졌을 때, 그 사용자의 특성을 다시 계산
  # 사용자가 평가한 데이터를 기반으로 해당 사용자를 모델에 추가하는 방식으로 동작

  # user_items : 사용자가 평가한 아이템 정보를 포함하는 희소 행렬
  # user_items 행렬에서 사용자가 평가한 아이템의 위치에 해당하는 데이터. 예를 들어, 사용자가 영화 1과 영화 5에 평점을 부여했을 경우, 해당 위치에 사용자의 평점이 기록된 행렬이 전달

  # recalculate_user=True : 사용자의 정보가 새롭게 주어졌을 때, 그 사용자의 특성을 다시 계산할지를 결정
  # 새로운 사용자나 현재까지 모델에 포함되지 않은 사용자에게 추천을 제공하기 위해 사용자의 특징 벡터를 새롭게 계산
  # 이 설정은 사용자의 기존 데이터를 학습한 모델에 반영하는 것이 아니라, 주어진 평가 정보(user_items)를 사용하여 그 사용자의 잠재적 특성을 즉시 추정하는 방식
  # 새로 추가된 사용자의 평가 데이터를 기반으로 모델이 해당 사용자의 잠재적 선호도를 추정하고, 이에 기반하여 추천을 생성

  # N=10 : 추천할 아이템의 개수. 여기서는 3개의 아이템을 추천
  recs = loaded_model.recommend(userid=0, user_items=user_items, recalculate_user=True, N=3)
  return [str(items[r]) for r in recs[0]] # 추천 결과를 아이템 아이디에서 아이템 이름으로 변환하여 반환

def build_matrix_input(input_rating_dict, items):
  model = pickle.load(open(saved_model_fname, 'rb'))


  item_ids = {r: i for i, r in items.items()} # item_ids : 아이템 ID를 인덱스로 변환하는 딕셔너리
  filtered_ratings = {s: input_rating_dict[s] for s in input_rating_dict if s in item_ids} # input_rating_dict에서 item_ids에 있는 아이템만 필터링


  # 필터링 후 데이터 생성
  mapped_idx = [item_ids[s] for s in filtered_ratings.keys()] # item_ids 딕셔너리를 사용하여 item_id를 인덱스로 변환
  data = [weight * float(x) for x in input_rating_dict.values()] # 가중치를 곱하여 데이터 생성
  # rows = [0 for _ in mapped_idx]
  rows = [0] * len(mapped_idx)  # rows는 항상 0으로 고정된 길이로 생성 이유는 사용자가 1명이기 때문에
  shape = (1, model.item_factors.shape[0])


  return coo_matrix((data, (rows, mapped_idx)), shape=shape).tocsr()

def user_based_recommendation(input_rating_dict):


  input_rating_dict = {int(k): v for k, v in input_rating_dict.items()} # input_rating_dict의 키와 값을 정수로 변환 : {"1": 3.5}와 같은 형식으로 전달 받았을 경우, {"1": 3.5}를 {1: 3.5}로 변환


  rating_df = pd.read_csv(ratings_fname)
  rating_df["userId"] = rating_df["userId"].astype("category")
  rating_df["movieId"] = rating_df["movieId"].astype("category")

  movie_df = pd.read_csv(final_fname)


  items = dict(enumerate(rating_df["movieId"].cat.categories))
  input_matrix = build_matrix_input(input_rating_dict, items)
  result = calculate_user_based(input_matrix, items)
  result = [int(x) for x in result]
  result_items = movie_df[movie_df["movieId"].isin(result)].to_dict("records")
  return json.dumps(result_items)


input_rating_dict = {
  "1":4,
  "2":3.5
}

if __name__ == "__main__":
  try:
    match sys.argv[1]:
      case "item-based":
        print(item_based_recommendation(int(sys.argv[2])))
      case "user-based":
        input_data = sys.stdin.read()
        print(user_based_recommendation(json.loads(input_data)))
      case _:
        print(json.dumps({"error": "Invalid command provided"}))
  except ValueError:
    print(json.dumps({"error": "Invalid arguments"}))
  # model_train() # 처음 한번 실행 후 주석처리. 이후 데이터가 추가되었을 때 주기적으로 실행

