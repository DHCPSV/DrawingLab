"""V4 API/scheduling tests; the scripted model is not image quality evidence."""
import copy
import threading
import time

import pytest
from fastapi.testclient import TestClient

from drawinglab.copaint import CopaintJobManager
from drawinglab.imaging import encode_png
from drawinglab.server import create_app
from test_smoke import MissingEngine, Response, ScriptedEngine, ink_image, request, wait_finished, project_temp


def stroke():
    return {"id":"editable", "tool":"pen", "size":3,
            "points":[{"x":100+i*4,"y":180+(i%2)*1.2} for i in range(26)]}


def body(task="cleanup"):
    s=stroke()
    ink=ink_image([(p["x"],p["y"]) for p in s["points"]])
    return {"revision":1,"intent_revision":0,"task":task,"prompt":"same character",
            "ink":encode_png(ink),"strokes":[s]}


def test_cleanup_available_without_model_and_does_not_mutate_source(project_temp):
    with TestClient(create_app(MissingEngine(),project_temp)) as client:
        sid=client.post('/api/sessions').json()['id']
        payload=body(); original=copy.deepcopy(payload)
        assert client.post(f'/api/sessions/{sid}/suggest',json=payload).status_code==202
        result=wait_finished(client.app.state.manager,sid,'ready')
        assert result['task']=='cleanup' and result['ghost'] is None
        assert result['metrics']['generated_by_ai'] is False
        assert result['raw_reference'] is None, 'CPU cleanup must not fabricate an AI original'
        assert result['correction']['stroke_updates']
        assert result['correction']['stroke_updates'][0]['original_points']==payload['strokes'][0]['points']
        assert result['correction']['metrics']['ai_matched_strokes']==0
        assert payload==original


class EditEngine(ScriptedEngine):
    supports_correction=True
    def generate_correction(self,**kwargs):
        kwargs['task_marker']='correction'
        return self.generate(**kwargs)


def test_ai_correction_uses_edit_engine_and_result_is_versioned(project_temp):
    engine=EditEngine(Response(ink_image([(100,183),(200,183)])))
    with TestClient(create_app(engine,project_temp)) as client:
        sid=client.post('/api/sessions').json()['id']
        payload=body('correction')
        assert client.post(f'/api/sessions/{sid}/suggest',json=payload).status_code==202
        result=wait_finished(client.app.state.manager,sid,'ready')
        assert engine.calls[0]['task_marker']=='correction'
        assert result['source_revision']==result['correction']['source_revision']==1
        assert result['correction']['intent_revision']==0
        assert result['ghost'] is None and result['correction']['stroke_updates']
        assert result['raw_reference']==encode_png(ink_image([(100,183),(200,183)]))
        client.post(f'/api/sessions/{sid}/invalidate',json={'revision':2,'intent_revision':1,'reset':True})
        assert client.get(f'/api/sessions/{sid}/result').json()['correction'] is None
        assert client.get(f'/api/sessions/{sid}/result').json()['raw_reference'] is None


def test_identical_completion_is_noop_not_rejected(project_temp):
    ink=ink_image([(100,100),(200,100)])
    manager=CopaintJobManager(ScriptedEngine(Response(ink)),project_temp)
    try:
        sid=manager.create(); manager.submit(sid,request(1,ink))
        result=wait_finished(manager,sid,'noop')
        assert result['metrics']['novel_pixels']==0
        assert result['metrics']['outcome']=='noop'
        assert result['ghost'] is None
        assert result['raw_reference']==encode_png(ink), 'No new ink must not hide the complete model image'
        assert '没有新增' in result['message']
    finally:manager.close()


def test_inflight_edit_discarded_after_new_ink(project_temp):
    gate=threading.Event(); engine=EditEngine(Response(ink_image([(100,183),(200,183)]),gate))
    manager=CopaintJobManager(engine,project_temp)
    try:
        sid=manager.create()
        manager.submit(sid,request(1,ink_image([(100,180),(200,180)]),task='correction',strokes=[stroke()]))
        engine.started.get(timeout=2)
        manager.invalidate(sid,2,1,True); gate.set()
        result=wait_finished(manager,sid,'idle')
        assert result['correction'] is None and result['ghost'] is None
        assert result['raw_reference'] is None
        assert result['revision']==2 and result['intent_revision']==1
    finally:gate.set(); manager.close()


@pytest.mark.parametrize('change',[
    {'region':{'x':750,'y':100,'width':50,'height':100}},
    {'region':{'x':100,'y':100,'width':0,'height':100}},
    {'locked_ids':['not-in-this-artwork']},
    {'strokes':[]},
])
def test_bad_correction_scope_is_rejected_at_boundary(project_temp,change):
    with TestClient(create_app(MissingEngine(),project_temp)) as client:
        sid=client.post('/api/sessions').json()['id']
        payload=dict(body(),**change)
        assert client.post(f'/api/sessions/{sid}/suggest',json=payload).status_code==422


def test_locked_artwork_returns_legitimate_no_change(project_temp):
    with TestClient(create_app(MissingEngine(),project_temp)) as client:
        sid=client.post('/api/sessions').json()['id']
        payload=dict(body(),locked_ids=['editable'])
        assert client.post(f'/api/sessions/{sid}/suggest',json=payload).status_code==202
        result=wait_finished(client.app.state.manager,sid,'noop')
        assert result['correction'] is None
        assert result['metrics']['correction']['locked_strokes']==1
