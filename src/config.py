"""경로 설정과 데이터 준비. Drive(데이터, 체크포인트, 결과)와 GitHub(코드)를 한 곳에서 이어준다.

팀원 모두 같은 코드로 같은 경로를 쓰도록 여기서만 경로를 정한다.
Drive 폴더 이름이 다르면 환경변수로 바꾼다 (코드 수정 불필요):
    import os; os.environ['CV_DRIVE_ROOT'] = '/content/drive/MyDrive/내폴더'

Drive 구조 (공유 폴더, 각자 '내 드라이브에 바로가기 추가'):
    CV_miniproject/
    ├── data/training_set.zip      # 원본 데이터 (읽기 전용으로 공유)
    ├── checkpoints/{model}/fold{k}.pth
    └── results/{method}__{split}__fold{k}.csv
GitHub에는 코드와 splits/*.csv만 둔다.
"""
import os
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(REPO_ROOT, "splits")  # 분할 파일은 GitHub가 유일한 원본

DRIVE_ROOT = os.environ.get("CV_DRIVE_ROOT", "/content/drive/MyDrive/CV_miniproject")
LOCAL_ROOT = os.environ.get("CV_LOCAL_ROOT", "/content/data")  # Colab 로컬 디스크 (빠름, 세션이 끊기면 사라짐)

DATA_ZIP = os.environ.get("CV_DATA_ZIP", os.path.join(DRIVE_ROOT, "data", "training_set.zip"))
DATA_DIR = os.path.join(LOCAL_ROOT, "training_set", "training_set")  # 압축을 풀면 폴더가 한 겹 중첩됨
HC_CSV = os.path.join(LOCAL_ROOT, "training_set_pixel_size_and_HC.csv")
HC_URL = "https://zenodo.org/records/1327317/files/training_set_pixel_size_and_HC.csv?download=1"

CKPT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
RESULT_DIR = os.path.join(DRIVE_ROOT, "results")


def prepare_data():
    """Drive의 zip을 Colab 로컬로 풀고 HC csv를 내려받는다. 이미 되어 있으면 건너뛴다.

    세션을 새로 켤 때마다 한 번 호출한다. (Drive에서 이미지를 직접 읽으면 학습이 매우 느려지므로 로컬에 푼다.)
    """
    os.makedirs(LOCAL_ROOT, exist_ok=True)

    if os.path.isdir(DATA_DIR) and any(f.endswith(".png") for f in os.listdir(DATA_DIR)):
        print("데이터가 이미 준비되어 있습니다:", DATA_DIR)
    else:
        if not os.path.exists(DATA_ZIP):
            raise FileNotFoundError(f"zip을 찾을 수 없습니다: {DATA_ZIP}\n"
                                    "Drive 마운트와 폴더 바로가기를 확인하거나 CV_DATA_ZIP를 설정하세요.")
        print("압축 해제 중:", DATA_ZIP)
        with zipfile.ZipFile(DATA_ZIP) as z:
            z.extractall(os.path.join(LOCAL_ROOT, "training_set"))
        print("완료:", DATA_DIR)

    if not os.path.exists(HC_CSV):
        try:
            urllib.request.urlretrieve(HC_URL, HC_CSV)
            print("HC csv 다운로드 완료")
        except Exception as e:  # HC 오차는 선택 항목이라 실패해도 진행
            print("HC csv 다운로드 실패 (HC 오차 계산은 건너뛸 수 있음):", e)


def ckpt_path(model_name, fold):
    """체크포인트 저장 경로 (Drive). 폴더가 없으면 만든다."""
    d = os.path.join(CKPT_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"fold{fold}.pth")
