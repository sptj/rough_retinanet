import os
import tensorflow as tf
from lxml import etree
import sys
def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def parse_voc_to_dict(annotation_file):
    path = os.path.join(annotation_file)
    with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = recursive_parse_xml_to_dict(xml)['annotation']
    return data
def voc_format_to_object_detect_format(data):
    filename=data['path']
    classes_text=[]
    box=[]
    boxes=[]
    if 'object' in data:
        for obj in data['object']:
            box.append(float(obj['bndbox']['xmin']))
            box.append(float(obj['bndbox']['ymin']))
            box.append(float(obj['bndbox']['xmax']))
            box.append(float(obj['bndbox']['ymax']))
            boxes.append(box[:])
            classes_text.append(obj['name'].encode('utf8'))
    return filename,classes_text, boxes

