from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from zipfile import ZipFile

import click
from loguru import logger
from PIL import Image, UnidentifiedImageError
from rarfile import RarFile
from tqdm import tqdm


def split_into_rows(image: Image.Image) -> list[Image.Image]:
    WHITE_THRESHOLD = 235
    PANEL_WIDTH_THRESHOLD = 30
    WHITE_PAGE_NUM_WHITE_ROWS_THRESHOLD = 0.4

    logger.debug(f"split_into_rows {image.size}")

    result = []
    image_bw = image.convert("L")

    left = 0
    right = image.size[0]

    y_avg_sum = [
        (
            sum(image_bw.getpixel((x, y)) for x in range(image_bw.size[0]))
            / image_bw.size[0]
        )
        for y in range(image_bw.size[1])
    ]
    num_white_rows = sum(1 for z in y_avg_sum if z >= WHITE_THRESHOLD)

    if num_white_rows / image.size[1] > WHITE_PAGE_NUM_WHITE_ROWS_THRESHOLD:
        return [image]

    y_prev_white = 0
    for y in range(image_bw.size[1]):
        if y_avg_sum[y] >= WHITE_THRESHOLD:
            if y - y_prev_white >= PANEL_WIDTH_THRESHOLD:
                upper = y_prev_white
                lower = y
                result.append(image.crop((left, upper, right, lower)))
                logger.debug(f"found panel {(left, upper, right, lower)}")
            y_prev_white = y
    if image.size[1] - y_prev_white >= PANEL_WIDTH_THRESHOLD:
        upper = y_prev_white
        lower = image.size[1]
        result.append(image.crop((left, upper, right, lower)))
        logger.debug(f"found panel {(left, upper, right, lower)}")

    return result


def split_into_cols(image: Image.Image) -> list[Image.Image]:
    # rotate image such that the leftmost panel becomes the uppermost
    logger.debug(f"split_into_cols {image.size}")
    return [
        panel.transpose(Image.ROTATE_90)
        for panel in split_into_rows(image.transpose(Image.ROTATE_270))
    ]


@dataclass
class ComicBookPage:
    image: list[Image.Image]
    filename: str

    @staticmethod
    def from_file(file: BinaryIO) -> "ComicBookPage | None":
        try:
            image = Image.open(file)
        except UnidentifiedImageError:
            logger.info(f"Could not load '{file.name}'")
            return None
        return ComicBookPage(image=image.convert("RGB"), filename=file.name)

    def save(self, file: BinaryIO) -> None:
        format = self.filename.split(".")[-1].upper()
        match format:
            case "JPG" | "JPEG":
                format = "JPEG"
            case _:
                raise ValueError(f"Unknown file format '{format}' to save")
        self.image.save(file, format)

    def split_panels(self) -> list["ComicBookPage"]:
        def get_new_filename(row, col):
            fields = self.filename.split(".")
            fields[-2] += f"_{row:04d}_{col:04d}"
            return ".".join(fields)

        logger.debug(f"split panels of {self.filename}")

        return [
            ComicBookPage(image=panel_final, filename=get_new_filename(row, col))
            for row, panel_row in enumerate(split_into_rows(self.image))
            for col, panel_final in enumerate(split_into_cols(panel_row))
        ]


@dataclass
class ComicBook:
    pages: list[ComicBookPage]

    @staticmethod
    def from_file(file: Path | RarFile | ZipFile) -> "ComicBook":
        match file:
            case Path():
                if file.suffix == ".cbr":
                    with RarFile(file) as cbr_file:
                        return ComicBook.from_file(cbr_file)
                elif file.suffix == ".cbz":
                    with ZipFile(file) as cbz_file:
                        return ComicBook.from_file(cbz_file)
                else:
                    raise ValueError(f"Unknown file format '{file.suffix}'")
            case RarFile() | ZipFile():
                return ComicBook(
                    pages=[
                        page
                        for name in sorted(file.namelist())
                        if not name.endswith("/")
                        and (page := ComicBookPage.from_file(file.open(name)))
                        is not None
                    ]
                )
            case _:
                raise ValueError(f"Illegal parameter type '{type(file)}'")

    def save(self, file: Path | RarFile | ZipFile) -> None:
        match file:
            case Path():
                if file.suffix == ".cbr":
                    with RarFile(file, "w") as cbr_file:
                        self.save(cbr_file)
                elif file.suffix == ".cbz":
                    with ZipFile(file, "w") as cbz_file:
                        self.save(cbz_file)
            case RarFile() | ZipFile():
                for page in self.pages:
                    with file.open(page.filename, "w") as page_file:
                        page.save(page_file)
            case _:
                raise ValueError(f"Parameter file of type '{type(file)}' not supported")

    def split_panels(self) -> "ComicBook":
        return ComicBook(
            pages=[
                page_final
                for page in tqdm(self.pages)
                for page_final in page.split_panels()
            ]
        )


@click.command()
@click.option("-i", "--input-file", type=click.Path(exists=True), required=True)
@click.option("-o", "--output-file", type=click.Path(writable=True), required=True)
def main(input_file: str, output_file: str) -> None:
    comic_book = ComicBook.from_file(Path(input_file))
    comic_book = comic_book.split_panels()
    comic_book.save(Path(output_file))
