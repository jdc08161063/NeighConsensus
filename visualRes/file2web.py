
import argparse
import os
import re

parser = argparse.ArgumentParser('Visualizing Training sample, top200 pairs from randomly top 2000 pairs')

parser.add_argument(
	'--outHtml', type=str, help='output html file')

parser.add_argument(
	'--imgDir', type=str, help='image directory')


args = parser.parse_args()



### Writing the table format###
f = open(args.outHtml, 'w')
f.write('<html>\n')
f.write('<head>\n')
f.write('\t<title></title>\n')
f.write('\t<meta name=\"keywords\" content= \"Visual Result\" />  <meta charset=\"utf-8\" />\n')
f.write('\t<meta name=\"robots\" content=\"index, follow\" />\n')
f.write('\t<meta http-equiv=\"Content-Script-Type\" content=\"text/javascript\" />\n')
f.write('\t<meta http-equiv=\"expires\" content=\"0\" />\n')
f.write('\t<meta name=\"description\" content= \"Project page of style.css\" />\n')
f.write('\t<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\" media=\"screen\" />\n')
f.write('\t<link rel=\"shortcut icon\" href=\"favicon.ico\" />\n')
f.write('</head>\n')
f.write('<body>\n')
f.write('<div id="website">\n')
f.write('<center>\n')
f.write('\t<div class=\"blank\"></div>\n')
f.write('\t<h1>\n')
f.write('\t</h1>\n')
f.write('</center>\n')
f.write('<div class=\"blank\"></div>\n')
f.write('<center>\n')
f.write('<div>\n')

f.write('</div>\n')

### ---HTML Table--- ###
f.write('<table>\n')
f.write('\t<tr>\n')
f.write('\t\t<th>No.</th>\n')
f.write('\t\t<th>Top5% Matches</th>\n')
f.write('\t\t<th>Top10% Matches </th>\n')
f.write('\t\t<th>Top20% Matches </th>\n')
f.write('\t\t<th>GT </th>\n')
f.write('\t\t<th>KeyPoint </th>\n')

f.write('\t</tr>\n')

nbFile = len(os.listdir(args.imgDir)) / 5

for j in  range(nbFile): 
	f.write('\t<tr>\n')
	
	msg = '\t\t<td>{:d}</td>\n'.format(j + 1)
	f.write(msg)

	
	imgPath = os.path.join(args.imgDir, 'Top5_{:d}.jpg'.format(j))
	msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a> </td>\n'.format(imgPath, imgPath, imgPath)
	f.write(msg)
	
	imgPath = os.path.join(args.imgDir, 'Top10_{:d}.jpg'.format(j))
	msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a> </td>\n'.format(imgPath, imgPath, imgPath)
	f.write(msg)
	
	imgPath = os.path.join(args.imgDir, 'Top20_{:d}.jpg'.format(j))
	msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a> </td>\n'.format(imgPath, imgPath, imgPath)
	f.write(msg)
	
	
	imgPath = os.path.join(args.imgDir, 'GT{:d}.jpg'.format(j))
	msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a> </td>\n'.format(imgPath, imgPath, imgPath)
	f.write(msg)
	
	imgPath = os.path.join(args.imgDir, 'KeyPointMatch{:d}.jpg'.format(j))
	msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a> </td>\n'.format(imgPath, imgPath, imgPath)
	f.write(msg)
	f.write('\t</tr>\n')

	
f.write('</table>\n')

f.write('</center>\n</div>\n </body>\n</html>\n')
f.close()



