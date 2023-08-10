import logging
import sys


def error_message_detail(error, error_detail: sys):
    _, _, tb = error_detail.exc_info()
    error_message = f"Error occured on line {tb.tb_lineno} of {tb.tb_frame.f_code.co_filename}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys) -> object:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


