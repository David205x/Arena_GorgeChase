import json
import numpy as np
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs


from .config import *


def load_map_meta(map_id):
    if map_id in MAP_CACHE:
        return MAP_CACHE[map_id]
    path = MAP_DIR / f"gorge_chase_map_{map_id}.json"
    meta = {"map_id": map_id, "name": f"map_{map_id}", "width": None, "height": None}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        meta.update(name=data.get("name", meta["name"]), width=data.get("width"), height=data.get("height"))
    MAP_CACHE[map_id] = meta
    return meta


class WebControlServer:
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT, logger=None):
        self.logger = logger
        self._cv = threading.Condition()
        self._pending_action = None
        self._trail = []
        self._explored = set()
        self._state = {
            "episode": 0,
            "step": 0,
            "done": False,
            "last_reward": 0.0,
            "status": "initializing",
            "obs": None,
            "ui": {},
            "action_range": [0, ACTION_COUNT - 1],
        }
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        if self.logger:
            self.logger.info(f"web control server started at http://{SERVER_HOST}:{SERVER_PORT}")

    def publish_obs(self, obs, episode, step, status, last_reward=0.0, done=False):
        with self._cv:
            if episode != self._state.get("episode"):
                self.reset_episode_history()
            obs_json = self._to_jsonable(obs)
            self._update_trail(obs_json)
            self._state.update(
                episode=episode,
                step=step,
                done=bool(done),
                last_reward=self._normalize_reward(last_reward),
                status=status,
                obs=obs_json,
                ui=self._build_ui_state(obs_json),
            )
            self._cv.notify_all()

    def wait_for_action(self):
        with self._cv:
            while self._pending_action is None:
                self._cv.wait()
            action = self._pending_action
            self._pending_action = None
            return action

    def submit_action(self, action):
        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"action must be in [0, {ACTION_COUNT - 1}]")
        with self._cv:
            self._pending_action = action
            self._state["status"] = f"action {action} submitted"
            self._cv.notify_all()

    def get_state(self):
        with self._cv:
            return dict(self._state)

    def _build_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    return self._send_html(outer._html())
                if self.path.startswith("/state"):
                    return self._send_json(outer.get_state())
                self.send_error(404, "Not Found")

            def do_POST(self):
                if not self.path.startswith("/action"):
                    self.send_error(404, "Not Found")
                    return
                content_length = int(self.headers.get("Content-Length", "0"))
                params = parse_qs(self.rfile.read(content_length).decode("utf-8"))
                action_text = params.get("action", [""])[0].strip()
                try:
                    action = int(action_text)
                    outer.submit_action(action)
                except Exception as exc:
                    return self._send_json({"ok": False, "error": str(exc)}, status=400)
                self._send_json({"ok": True, "action": action, "state": outer.get_state()})

            def log_message(self, format, *args):
                if outer.logger:
                    outer.logger.info("[web] " + format % args)

            def _send_json(self, payload, status=200):
                body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, html, status=200):
                body = html.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _html(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        return template.replace("__ACTION_GROUPS__", json.dumps(ACTION_GROUPS, ensure_ascii=False))

    def _build_ui_state(self, obs):
        observation = obs.get("observation") or {}
        frame_state = observation.get("frame_state") or {}
        env_info = observation.get("env_info") or {}
        extra_info = obs.get("extra_info") or {}
        hero = frame_state.get("heroes") or {}
        hero_pos = hero.get("pos") or {}
        local_map = observation.get("map_info") or []
        visible_organs = frame_state.get("organs") or []
        monsters = frame_state.get("monsters") or []
        map_id = extra_info.get("map_id")
        size = len(local_map)
        center = size // 2 if size else 10
        trail_lookup = self._trail_lookup(hero_pos, center, size)
        organ_lookup = self._organ_lookup(hero_pos, visible_organs, center, size)
        monster_lookup = self._monster_lookup(hero_pos, monsters, center, size)
        ui_map = []
        for r, row in enumerate(local_map):
            ui_row = []
            for c, cell in enumerate(row):
                tile = {"terrain": int(cell), "entity": None, "label": "", "trail": (r, c) in trail_lookup}
                if r == center and c == center:
                    tile.update(entity="hero", label="我", trail=True)
                elif (r, c) in monster_lookup:
                    tile.update(entity="monster", label=monster_lookup[(r, c)])
                elif (r, c) in organ_lookup:
                    tile.update(**organ_lookup[(r, c)])
                ui_row.append(tile)
            ui_map.append(ui_row)
        return {
            "map_meta": load_map_meta(map_id) if map_id is not None else None,
            "hero": {"hero_id": hero.get("hero_id"), "pos": hero_pos, "flash_cooldown": hero.get("flash_cooldown"), "buff_remaining_time": hero.get("buff_remaining_time")},
            "summary": {"result_code": extra_info.get("result_code"), "result_message": extra_info.get("result_message"), "step_no": observation.get("step_no"), "max_step": env_info.get("max_step"), "total_score": env_info.get("total_score"), "step_score": env_info.get("step_score"), "treasure_score": env_info.get("treasure_score"), "treasures_collected": env_info.get("treasures_collected"), "total_treasure": env_info.get("total_treasure"), "collected_buff": env_info.get("collected_buff"), "total_buff": env_info.get("total_buff"), "flash_count": env_info.get("flash_count")},
            "monsters": monsters,
            "visible_organs": visible_organs,
            "legal_action": observation.get("legal_action") or [],
            "local_map": ui_map,
            "minimap": self._build_minimap(map_id, hero_pos, size),
            "trail": list(self._trail),
            "raw_preview": json.dumps(obs, ensure_ascii=False, indent=2)[:3000],
        }

    def reset_episode_history(self):
        self._trail = []
        self._explored = set()

    def _update_trail(self, obs):
        hero_pos = ((obs.get("observation") or {}).get("frame_state") or {}).get("heroes", {}).get("pos") or {}
        x, z = hero_pos.get("x"), hero_pos.get("z")
        if x is None or z is None:
            return
        point = {"x": int(x), "z": int(z)}
        self._explored.add((point["x"], point["z"]))
        if not self._trail or self._trail[-1] != point:
            self._trail.append(point)
        self._trail = self._trail[-64:]

    def _trail_lookup(self, hero_pos, center, size):
        hx, hz = hero_pos.get("x"), hero_pos.get("z")
        if hx is None or hz is None:
            return set()
        lookup = set()
        for point in self._trail:
            row = center + int(point["z"]) - int(hz)
            col = center + int(point["x"]) - int(hx)
            if 0 <= row < size and 0 <= col < size:
                lookup.add((row, col))
        return lookup

    def _build_minimap(self, map_id, hero_pos, local_size):
        if map_id is None:
            return None
        path = MAP_DIR / f"gorge_chase_map_{map_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        width = int(data.get("width") or 0)
        height = int(data.get("height") or 0)
        cells = [[0 for _ in range(width)] for _ in range(height)]
        for cell in data.get("cells", []):
            x = int(cell.get("x", 0))
            z = int(cell.get("z", 0))
            if 0 <= z < height and 0 <= x < width:
                cells[z][x] = int(cell.get("type_id", 0))
        hx = int(hero_pos.get("x", 0)) if hero_pos else 0
        hz = int(hero_pos.get("z", 0)) if hero_pos else 0
        half = local_size // 2 if local_size else 10
        viewport = {"x": max(hx - half, 0), "z": max(hz - half, 0), "w": local_size or 21, "h": local_size or 21}
        explored = [{"x": x, "z": z} for x, z in sorted(self._explored)]
        return {"cells": cells, "hero": {"x": hx, "z": hz}, "viewport": viewport, "trail": list(self._trail), "explored": explored}

    def _organ_lookup(self, hero_pos, organs, center, size):
        lookup = {}
        hx, hz = hero_pos.get("x"), hero_pos.get("z")
        if hx is None or hz is None:
            return lookup
        for organ in organs:
            pos = organ.get("pos") or {}
            row = center + int(pos.get("z", hz)) - int(hz)
            col = center + int(pos.get("x", hx)) - int(hx)
            if 0 <= row < size and 0 <= col < size:
                treasure = organ.get("sub_type") == 1
                lookup[(row, col)] = {"entity": "treasure" if treasure else "buff", "label": f"{'T' if treasure else 'B'}{organ.get('config_id', '')}"}
        return lookup

    def _monster_lookup(self, hero_pos, monsters, center, size):
        lookup = {}
        hx, hz = hero_pos.get("x"), hero_pos.get("z")
        if hx is None or hz is None:
            return lookup
        for idx, monster in enumerate(monsters, start=1):
            if monster.get("is_in_view") != 1:
                continue
            pos = monster.get("pos") or {}
            row = center + int(pos.get("z", hz)) - int(hz)
            col = center + int(pos.get("x", hx)) - int(hx)
            if 0 <= row < size and 0 <= col < size:
                lookup[(row, col)] = f"M{idx}"
        return lookup

    def _to_jsonable(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return repr(value)

    def _normalize_reward(self, reward):
        if isinstance(reward, dict):
            reward_value = reward.get("reward", reward)
            if isinstance(reward_value, dict):
                return self._to_jsonable(reward_value)
            try:
                return float(reward_value)
            except (TypeError, ValueError):
                return self._to_jsonable(reward)
        try:
            return float(reward)
        except (TypeError, ValueError):
            return self._to_jsonable(reward)
