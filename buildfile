./: {*/ -build/ -tools/ -certifiable-build/ -include/ -.github/ -docs/} manifest

./: src/ (examples/) tests/

if ($c.target.system == 'linux')
  m = lib{m}

import src = src/
import examples = examples/
import tests = tests/
