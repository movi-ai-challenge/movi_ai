from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd


# ============================================================
# config import
#
# 1. python -m src.fraud_detection.data_loader
# 2. VSCode에서 data_loader.py 직접 실행
#
# 두 방식 모두 지원하기 위한 처리
# ============================================================

try:
    from .config import (
        CSV_ENCODING,
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
    )
except ImportError:
    from config import (
        CSV_ENCODING,
        ELECTRONIC_CONFIG,
        CARD_CONFIG,
    )


# ============================================================
# 1. 데이터셋 설정
# ============================================================

DATASET_CONFIGS = {
    "electronic": ELECTRONIC_CONFIG,
    "card": CARD_CONFIG,
}


# ============================================================
# 2. Dataset Config 반환
# ============================================================

def get_dataset_config(dataset_type: str) -> dict:
    """
    데이터 종류에 맞는 config를 반환한다.

    Parameters
    ----------
    dataset_type : str
        "electronic" 또는 "card"

    Returns
    -------
    dict
        해당 데이터셋의 설정값
    """

    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(
            f"지원하지 않는 dataset_type입니다: {dataset_type}\n"
            f"사용 가능한 값: {list(DATASET_CONFIGS.keys())}"
        )

    return DATASET_CONFIGS[dataset_type]


# ============================================================
# 3. Train / Validation 폴더 반환
# ============================================================

def get_data_directory(
    dataset_type: str,
    split: str,
) -> Path:
    """
    데이터셋 종류와 split에 따라 실제 폴더 경로를 반환한다.

    Examples
    --------
    electronic + train
        -> data/Train/TL_전자금융공동망

    card + validation
        -> data/Validation/VL_카드거래
    """

    config = get_dataset_config(dataset_type)

    split = split.lower()

    if split == "train":
        directory = config["train_dir"]

    elif split in ("validation", "val"):
        directory = config["validation_dir"]

    else:
        raise ValueError(
            f"지원하지 않는 split입니다: {split}\n"
            "사용 가능한 값: train, validation"
        )

    if not directory.exists():
        raise FileNotFoundError(
            f"데이터 폴더가 존재하지 않습니다.\n"
            f"path: {directory}"
        )

    return directory


# ============================================================
# 4. CSV 파일 탐색
# ============================================================

def find_csv_files(
    directory: Path,
) -> list[Path]:
    """
    지정된 디렉토리 내부의 모든 CSV 파일을 찾는다.

    하위 폴더가 있는 경우에도 탐색하도록 rglob을 사용한다.
    """

    csv_files = sorted(
        file
        for file in directory.rglob("*.csv")
        if file.is_file()
    )

    if not csv_files:
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다.\n"
            f"path: {directory}"
        )

    return csv_files


# ============================================================
# 5. 컬럼 구조 검증
# ============================================================

def validate_columns(
    reference_columns: Sequence[str],
    current_columns: Sequence[str],
    file_path: Path,
) -> None:
    """
    여러 분기의 CSV 파일이 동일한 컬럼 구조를 가지는지 검사한다.

    컬럼이 다르면 pd.concat 과정에서 NaN 컬럼이 생길 수 있으므로
    조기에 오류를 발생시킨다.
    """

    reference_columns = list(reference_columns)
    current_columns = list(current_columns)

    if reference_columns == current_columns:
        return

    reference_set = set(reference_columns)
    current_set = set(current_columns)

    missing_columns = reference_set - current_set
    extra_columns = current_set - reference_set

    raise ValueError(
        "\nCSV 컬럼 구조가 서로 다릅니다.\n"
        f"문제 파일: {file_path}\n"
        f"누락 컬럼: {sorted(missing_columns)}\n"
        f"추가 컬럼: {sorted(extra_columns)}"
    )


# ============================================================
# 6. 여러 CSV를 하나의 DataFrame으로 Load
# ============================================================

def load_csv_directory(
    directory: Path,
    *,
    encoding: str = CSV_ENCODING,
    usecols: Optional[Sequence[str]] = None,
    nrows_per_file: Optional[int] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    특정 폴더에 존재하는 여러 CSV를 읽어서 하나의 DataFrame으로 병합한다.

    Parameters
    ----------
    directory : Path
        CSV 파일이 존재하는 폴더

    encoding : str
        CSV 인코딩

    usecols : list[str] | None
        필요한 컬럼만 읽고 싶은 경우 지정

    nrows_per_file : int | None
        각 CSV에서 몇 행만 읽을지 지정.
        테스트할 때 사용한다.

    max_files : int | None
        최대 몇 개의 CSV 파일을 읽을지 지정.
        테스트할 때 사용한다.

    Returns
    -------
    pd.DataFrame
        모든 CSV가 합쳐진 DataFrame
    """

    csv_files = find_csv_files(directory)

    # 테스트 시 일부 파일만 사용
    if max_files is not None:
        csv_files = csv_files[:max_files]

    print("=" * 60)
    print(f"[DATA LOADER] 폴더: {directory}")
    print(f"[DATA LOADER] CSV 파일 수: {len(csv_files)}")
    print("=" * 60)

    dataframes = []

    reference_columns = None

    for index, file_path in enumerate(csv_files, start=1):

        print(
            f"[{index}/{len(csv_files)}] "
            f"Loading: {file_path.name}"
        )

        df = pd.read_csv(
            file_path,
            encoding=encoding,
            usecols=usecols,
            nrows=nrows_per_file,
            low_memory=False,
        )

        # 첫 번째 CSV의 컬럼 구조 저장
        if reference_columns is None:
            reference_columns = df.columns.tolist()

        # 두 번째 CSV부터 컬럼 동일 여부 검사
        else:
            validate_columns(
                reference_columns,
                df.columns.tolist(),
                file_path,
            )

        dataframes.append(df)

    # ========================================================
    # 모든 분기 데이터 결합
    # ========================================================

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
        copy=False,
    )

    print()
    print("=" * 60)
    print("[DATA LOADER] Load 완료")
    print(f"행 개수: {len(combined_df):,}")
    print(f"열 개수: {len(combined_df.columns)}")
    print("=" * 60)

    return combined_df


# ============================================================
# 7. Dataset 단위 Load
# ============================================================

def load_dataset(
    dataset_type: str,
    split: str,
    *,
    usecols: Optional[Sequence[str]] = None,
    nrows_per_file: Optional[int] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    dataset_type과 split만 지정하면 데이터를 읽어오는
    메인 인터페이스.

    Examples
    --------

    전자금융 Train 전체:

        df = load_dataset(
            "electronic",
            "train"
        )

    카드 Validation:

        df = load_dataset(
            "card",
            "validation"
        )
    """

    directory = get_data_directory(
        dataset_type,
        split,
    )

    return load_csv_directory(
        directory,
        usecols=usecols,
        nrows_per_file=nrows_per_file,
        max_files=max_files,
    )


# ============================================================
# 8. 대용량 데이터 Chunk Loader
# ============================================================

def iter_dataset_chunks(
    dataset_type: str,
    split: str,
    *,
    chunksize: int = 100_000,
    usecols: Optional[Sequence[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    대용량 CSV를 한 번에 메모리에 올리지 않고
    chunksize 단위로 읽는다.

    이후 train_iforest.py에서 정상거래를 Sampling할 때 사용한다.

    Parameters
    ----------
    dataset_type : str
        electronic / card

    split : str
        train / validation

    chunksize : int
        한 번에 읽을 행 수

    usecols : list[str] | None
        필요한 컬럼만 읽을 경우 지정

    Yields
    ------
    pd.DataFrame
        chunksize 크기의 DataFrame
    """

    directory = get_data_directory(
        dataset_type,
        split,
    )

    csv_files = find_csv_files(directory)

    print("=" * 60)
    print("[CHUNK LOADER]")
    print(f"Dataset : {dataset_type}")
    print(f"Split   : {split}")
    print(f"Files   : {len(csv_files)}")
    print(f"Chunk   : {chunksize:,}")
    print("=" * 60)

    for file_index, file_path in enumerate(
        csv_files,
        start=1,
    ):

        print(
            f"[{file_index}/{len(csv_files)}] "
            f"Processing: {file_path.name}"
        )

        chunk_reader = pd.read_csv(
            file_path,
            encoding=CSV_ENCODING,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )

        for chunk in chunk_reader:
            yield chunk


# ============================================================
# 9. 데이터 기본 정보 출력
# ============================================================

def print_dataset_summary(
    df: pd.DataFrame,
    target_column: str = "이상거래여부",
) -> None:
    """
    로딩한 데이터의 기본적인 상태를 확인한다.
    """

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print()
    print("[Columns]")

    for column in df.columns:
        print(
            f"- {column}: {df[column].dtype}"
        )

    # ========================================================
    # Target 분포
    # ========================================================

    if target_column in df.columns:

        print()
        print("[Target Distribution]")

        counts = (
            df[target_column]
            .value_counts(dropna=False)
            .sort_index()
        )

        print(counts)

        print()
        print("[Target Ratio]")

        ratios = (
            df[target_column]
            .value_counts(
                normalize=True,
                dropna=False,
            )
            .sort_index()
        )

        print(
            (ratios * 100)
            .round(4)
            .astype(str)
            + "%"
        )

    print("=" * 60)


# ============================================================
# 10. 테스트
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 전체 400만 건을 바로 읽지 않는다.
    #
    # CSV 2개에서 각각 1,000행씩만 읽어서
    # Loader가 정상 동작하는지 테스트한다.
    # --------------------------------------------------------

    electronic_test = load_dataset(
        dataset_type="electronic",
        split="train",
        nrows_per_file=1_000,
        max_files=2,
    )

    print_dataset_summary(
        electronic_test
    )

    print()
    print("[HEAD]")
    print(electronic_test.head())

'''
for chunk in iter_dataset_chunks(
    "electronic",
    "train",
    chunksize=100_000,
):
    print(chunk.shape)
'''