"""Reference precedence checks, independent of real model quality."""
from PIL import Image, ImageDraw

from drawinglab.klein_engine import prepare_references


def sketch():
    image = Image.new('RGB', (768, 768), 'white')
    ImageDraw.Draw(image).line([(300, 200), (300, 400), (400, 450)], fill='black', width=3)
    return image


def test_current_ink_is_the_only_style_reference_without_local_context():
    ink = sketch()
    references, metadata = prepare_references(ink, sketch(), None, 768)
    assert len(references) == 1
    assert references[0].tobytes() == ink.tobytes()
    assert metadata['style_source'] == 'current_user_ink_only'
    assert metadata['previous_reference_used'] is False


def test_small_compatible_reference_has_changed_area_removed_without_mutating_inputs():
    ink = sketch()
    prior = ink.copy()
    ImageDraw.Draw(prior).line([(600, 600), (650, 650)], fill='black', width=3)
    old_bytes = prior.tobytes()
    mask = Image.new('L', ink.size, 0)
    ImageDraw.Draw(mask).rectangle((570, 570, 680, 680), fill=255)
    references, metadata = prepare_references(ink, prior, mask, 768)
    assert len(references) == 2
    assert metadata['previous_reference_used'] is True
    assert references[0].tobytes() == ink.tobytes()
    assert references[1].getpixel((620, 620)) == (255, 255, 255)
    assert prior.tobytes() == old_bytes


def test_existing_ink_disagreement_discards_previous_hypothesis():
    ink = sketch()
    prior = Image.new('RGB', ink.size, 'white')
    ImageDraw.Draw(prior).line([(500, 200), (500, 400)], fill='black', width=3)
    mask = Image.new('L', ink.size, 0)
    ImageDraw.Draw(mask).rectangle((270, 370, 350, 480), fill=255)
    references, metadata = prepare_references(ink, prior, mask, 768)
    assert len(references) == 1
    assert metadata['previous_ink_mismatch_fraction'] > 0.05
    assert metadata['previous_reference_used'] is False


def test_global_edit_discards_previous_hypothesis_and_flattens_transparency():
    ink = Image.new('RGBA', (768, 768), (0, 0, 0, 0))
    ImageDraw.Draw(ink).line([(300, 200), (300, 400)], fill=(0, 0, 0, 255), width=3)
    references, metadata = prepare_references(ink, sketch(), Image.new('L', ink.size, 255), 512)
    assert len(references) == 1
    assert references[0].size == (512, 512)
    assert references[0].getpixel((0, 0)) == (255, 255, 255)
    assert metadata['edit_fraction'] == 1
    assert metadata['previous_reference_used'] is False
