from pathlib import Path

# ========== server address
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 18081
# ========== web front
TEMPLATE_PATH = Path(__file__).resolve().with_name('web_console.html')
# ========== map
MAP_DIR = Path(__file__).resolve().parents[1] / 'ref'
MAP_CACHE = {}

# ========== web action display
ACTION_COUNT = 16
ACTION_GROUPS = [
    {
        'title': '移动',
        'kind': 'move',
        'slots': [
            {'id': 3, 'name': '西北', 'arrow': '↖'},
            {'id': 2, 'name': '北', 'arrow': '↑'},
            {'id': 1, 'name': '东北', 'arrow': '↗'},
            {'id': 4, 'name': '西', 'arrow': '←'},
            None,
            {'id': 0, 'name': '东', 'arrow': '→'},
            {'id': 5, 'name': '西南', 'arrow': '↙'},
            {'id': 6, 'name': '南', 'arrow': '↓'},
            {'id': 7, 'name': '东南', 'arrow': '↘'},
        ],
    },
    {
        'title': '闪现',
        'kind': 'flash',
        'slots': [
            {'id': 11, 'name': '闪西北', 'arrow': '⇖'},
            {'id': 10, 'name': '闪北', 'arrow': '⇧'},
            {'id': 9, 'name': '闪东北', 'arrow': '⇗'},
            {'id': 12, 'name': '闪西', 'arrow': '⇐'},
            None,
            {'id': 8, 'name': '闪东', 'arrow': '⇒'},
            {'id': 13, 'name': '闪西南', 'arrow': '⇙'},
            {'id': 14, 'name': '闪南', 'arrow': '⇓'},
            {'id': 15, 'name': '闪东南', 'arrow': '⇘'},
        ],
    },
]