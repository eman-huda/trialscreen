# Error-analysis worksheet (test split, fine-tuned model)

87 of 206 predictions differ from the expert label or evidence set.


## over-confident: decided despite missing information (28)

- annotation 169 (sigir-201410, NCT02097186, exclusion): criterion: 'Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2' | expert: not enough information [] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'
- annotation 174 (sigir-201410, NCT02097186, exclusion): criterion: 'Patients with previous axillary surgery' | expert: not enough information [] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'
- annotation 193 (sigir-201411, NCT01526382, exclusion): criterion: 'Patients were moribund.' | expert: not enough information [0, 1, 2, 3, 4] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'
- annotation 315 (sigir-201417, NCT01745731, exclusion): criterion: 'BMI> 40 kg/m2.' | expert: not enough information [] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'

## right label, partial evidence (20)

- annotation 166 (sigir-201410, NCT02097186, exclusion): criterion: 'Significant upper limb peripheral arterial disease' | expert: not excluded [] | model: not excluded [2] | raw: '{"evidence": [2], "label": "not excluded"}'
- annotation 189 (sigir-201411, NCT01526382, inclusion): criterion: 'at least 1 of 3 signs of hypoperfusion (urine output < 0.5mL/kg of body weight per hour for 1 hour or more; neurologic dysfunction defined by confusion, psychos' | expert: not enough information [] | model: not enough information [2] | raw: '{"evidence": [2], "label": "not enough information"}'
- annotation 307 (sigir-201417, NCT01745731, inclusion): criterion: 'Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary' | expert: not enough information [3] | model: not enough information [0, 1, 2, 3, 4] | raw: '{"evidence": [0, 1, 2, 3, 4], "label": "not enough information"}'
- annotation 309 (sigir-201417, NCT01745731, exclusion): criterion: 'Different tumor records current disease or any disease hematologic.' | expert: not excluded [] | model: not excluded [0] | raw: '{"evidence": [0], "label": "not excluded"}'

## missed available evidence (predicted not enough information) (14)

- annotation 161 (sigir-201410, NCT02097186, inclusion): criterion: 'Patients undergoing elective carotid endarterectomy' | expert: not included [] | model: not enough information [0, 1, 2] | raw: '{"evidence": [0, 1, 2], "label": "not enough information"}'
- annotation 162 (sigir-201410, NCT02097186, inclusion): criterion: 'Patients undergoing open abdominal aortic aneurysm repair' | expert: not included [] | model: not enough information [0, 1, 2] | raw: '{"evidence": [0, 1, 2], "label": "not enough information"}'
- annotation 163 (sigir-201410, NCT02097186, inclusion): criterion: 'Patients undergoing endovascular abdominal aneurysm repair' | expert: not included [] | model: not enough information [0, 1, 2] | raw: '{"evidence": [0, 1, 2], "label": "not enough information"}'
- annotation 177 (sigir-201410, NCT02264964, inclusion): criterion: 'Maintenance hemodialysis after central vena catheterization.' | expert: not included [] | model: not enough information [] | raw: '{"evidence": [], "label": "not enough information"}'

## right label, wrong evidence (14)

- annotation 176 (sigir-201410, NCT02264964, inclusion): criterion: 'No medical history of central vena catheterization.' | expert: included [0] | model: included [] | raw: '{"evidence": [], "label": "included"}'
- annotation 179 (sigir-201410, NCT02264964, exclusion): criterion: 'Had been performed central venous puncture or catheterization before.' | expert: not excluded [0] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'
- annotation 197 (sigir-201411, NCT01723137, exclusion): criterion: 'Decreased level of consciousness' | expert: not excluded [0, 5] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'
- annotation 319 (sigir-201417, NCT01745731, exclusion): criterion: 'Difficulty in monitoring.' | expert: not excluded [5] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'

## unsupported decision (no evidence cited) (9)

- annotation 322 (sigir-201417, NCT01745731, exclusion): criterion: 'Pregnant women or women of childbearing age who do not have adequate contraception.' | expert: not excluded [0] | model: not applicable [] | raw: '{"evidence": [], "label": "not applicable"}'
- annotation 683 (sigir-20158, NCT02562040, exclusion): criterion: 'Previous tonsillectomy, including partial tonsillectomy' | expert: not applicable [] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'
- annotation 693 (sigir-20158, NCT00393913, exclusion): criterion: 'Suspected diagnosis of a sleep disorder other than OSA (i.e., periodic leg movements, narcolepsy, insomnia, central sleep apnea, sleep hypoventilation syndrome)' | expert: not applicable [] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'
- annotation 697 (sigir-20158, NCT00393913, exclusion): criterion: 'Pregnant' | expert: not applicable [] | model: not excluded [] | raw: '{"evidence": [], "label": "not excluded"}'

## contradictory polarity (1)

- annotation 164 (sigir-201410, NCT02097186, inclusion): criterion: 'Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)' | expert: not included [] | model: included [0, 2] | raw: '{"evidence": [0, 2], "label": "included"}'

## applicability confusion (1)

- annotation 767 (sigir-201513, NCT00315042, exclusion): criterion: 'Symptoms that collectively suggest nonstreptococcal T/P (eg, laryngitis, coryza, conjunctivitis, diarrhea, cough)' | expert: not applicable [0] | model: not excluded [0, 1, 2] | raw: '{"evidence": [0, 1, 2], "label": "not excluded"}'