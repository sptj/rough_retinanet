import tensorflow as tf
from lxml import etree
import os
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

def parse_voc_to_dict(filename):
    def tf_parse_voc_to_dict(filename):
        with tf.gfile.GFile(filename,'r') as  fid:
            annotation_string=fid.read()
        annotation_tree=etree.fromstring(annotation_string)
        parsed_annotations=recursive_parse_xml_to_dict(annotation_tree)
        return parsed_annotations['annotation']
    def bald_parse_voc_to_dict(filename):
        pass
    return tf_parse_voc_to_dict(filename)


def voc_format_to_object_detect_format(data,normalized=False):
    boxes=[]
    classes_text = []
    filename=data['filename']
    _, suffix = os.path.splitext(filename)
    if suffix == '':
        _,filename= os.path.split(data['path'])


    width = int(data['size']['width'])
    height = int(data['size']['height'])

    for obj in data['object']:
        if normalized:
            xmin=(float(obj['bndbox']['xmin']) / width)
            ymin=(float(obj['bndbox']['ymin']) / height)
            xmax=(float(obj['bndbox']['xmax']) / width)
            ymax=(float(obj['bndbox']['ymax']) / height)
        else:
            xmin=(float(obj['bndbox']['xmin']) )
            ymin=(float(obj['bndbox']['ymin']) )
            xmax=(float(obj['bndbox']['xmax']) )
            ymax=(float(obj['bndbox']['ymax']) )

        classes_text.append(obj['name'].encode('utf8'))
        boxes.append([xmin,ymin,xmax,ymax])

        return filename,classes_text,boxes






if __name__ == '__main__':
    ANNOTATION_FILENAME = 'F:\\data_set\\1.xml'
    temp=parse_voc_to_dict(ANNOTATION_FILENAME)
    print(temp)
    print(voc_format_to_object_detect_format(temp))


