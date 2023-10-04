
import json
import os
import pickle
codecs = ['CP949', 'utf8', 'euc-kr',
          'ascii', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855',
          'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949',
          'cp950', 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256', 'cp1257', 'cp1258',
          'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr', 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
          'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6',
          'iso8859_7', 'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t',
          'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004',
          'shift_jisx0213', 'utf_32', 'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_8', 'utf_8_sig']


def read(file_path, d1, condition_record, c_sites, p=False):
    filetype = file_path.split(".")[-1].lower()
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            if filetype == "json":
                return json.load(f)
            elif filetype =="csv":
                for codec in codecs:
                    try:
                        csv_df = pd.read_csv(file_path
                                             , header=0
                                             , encoding=codec
                                             # , sep=',|\s+|;|:'
                                             , sep=','
                                             # , iterator=True, chunksize=10000 # b. -1~0% (73~74초 → 74초), 파일 사이즈가 큰 경우 유리
                                             # , dtype=None
                                             # , usecols = cols
                                             , )
                    except:
                        continue
                    csv_df = condition(columnControl(csv_df, d1), condition_record, p=p)

                    return csv_df

                raise Exception("Encoding Error for {}".format(file_path))

            elif filetype == "mat":
                from scipy import io
                return io.loadmat(file_path)
            elif filetype == "xml":
                pass
            elif filetype == "pickle":
                return pickle.load(f)
    else:
        return False