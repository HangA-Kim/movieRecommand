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



if __name__ == "__main__":
  try:
    match sys.argv[1]:
      case "item-based":
        print(item_based_recommendation(int(sys.argv[2])))
      case _:
        print(json.dumps({"error": "Invalid command provided"}))
  except ValueError:
    print(json.dumps({"error": "Invalid arguments"}))
  # model_train() # 처음 한번 실행 후 주석처리. 이후 데이터가 추가되었을 때 주기적으로 실행

