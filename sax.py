import xml.sax
import sys

class handler(xml.sax.ContentHandler):
  def startDocument(self):
    print("start")
  def endDocument(self):
    print("end")
  def startElement(self, tag, attrs):
    print("start", tag)
  def endElement(self, tag):
    print("end", tag)
  def characters(self, data):
    print("chars    ", data.strip())

h = handler()

p = xml.sax.parse(sys.stdin, h)
