import requests
import re
from urllib.parse import unquote
import os

def download_icons(debug_show=False):
    icon_path = './icon'
    if not os.path.exists(icon_path):
        os.makedirs(icon_path)

    url = "https://prts.wiki/w/%E5%90%8E%E5%8B%A4%E6%8A%80%E8%83%BD%E4%B8%80%E8%A7%88"
    resp = requests.get(url)

    if resp.status_code != 200:
        raise RuntimeError("Failed to catch prts wiki page")

    content = resp.content.decode('utf-8')

    res = re.findall(r'data-src="/images/.+/.+/Bskill.+png', content, flags=0)
    for r in res:
        download_url = str(r).replace('data-src=\"', r'https://prts.wiki')
        filename = unquote(download_url.split('/')[-1])
        file_path = os.path.join(icon_path, filename)
        if os.path.exists(file_path):
            continue
        resp = requests.get(download_url)
        if resp.status_code != 200:
            raise RuntimeError("Failed to download icon" + filename)
        with open(file_path, 'wb') as f:
            f.write(resp.content)

        if debug_show:
            print(filename)

if __name__ == '__main__':
    download_icons()
