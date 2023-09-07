import argparse
import yaml
from kgx.graph.base_graph import BaseGraph
from kgx.graph.nx_graph import NxGraph
import uuid
import pandas as pd
from kgx.transformer import Transformer
from os import path

def load_config(descriptor):
    with open(descriptor, 'r') as stream:
        return yaml.safe_load(stream)

CONFIG_FILENAME = path.join(path.dirname(path.abspath(__file__)), "global-map.yaml")
def load_global_map(global_map_file: str = CONFIG_FILENAME):
    """Loads the global map yaml file and returns the dictionary.
    Parameters
    ----------
    global_map_file: str
        The filename with all the global map

    Returns
    -------
    Dict
        A Dictionary containing all the entries from the global map YAML
    
    """
    with open(global_map_file, 'r') as stream:
        return yaml.safe_load(stream)
        

class DemKGTransformer:
    
    def __init__(self, descriptor='data-descriptor.yaml'):
        self.descriptor = load_config(descriptor)
        self.global_map = load_global_map()
        self.graph = NxGraph()

    def config(self, field):
        """
        Returns the value of the specified field from the descriptor's 'config' dictionary.

        Parameters
        ----------
            field (str): The name of the field to retrieve.

        Returns
        -------
            The value of the specified field.
        """
        return self.descriptor['config'][field]

    def transform(self):
        """
        Transforms the input data file according to the configuration specified in the descriptor.
        Reads the input file in the format specified in the descriptor, initializes the visits dictionary,
        and applies the __process_row method to each row of the input data.
        """
        input_file = self.descriptor['config']['iput']('file')
        input_format = self.descriptor['config']['iput']('format')
        self.prefix = self.descriptor['config']['prefix']
        
        sep  = ','
        if input_format == 'tsv':
            sep = '\t'

        self.data = pd.read_csv(input_file, sep=sep, header=0)
        self.__init_visits(self.data)
        self.data.apply(self.__process_row, axis=1)

    def save(self):
        transformer = Transformer()
        input_args = {
            "format": "graph", 
            "graph": self.graph,
            }
        filename = self.descriptor['config']['destination']
        output_args = {
            'filename': filename,
            'format': 'tsv',
            'edge_properties':{"subject", "predicate", "object", "relation"}
        }
        output_args = {
            'filename': filename,
            'format': 'tsv',
            'edge_properties':{"id", "subject", "predicate", "object", "relation"}
        }

        transformer.transform(input_args)   
        transformer.save(output_args)

    def __process_row(self, row):
        self.__read_subject(row)
        self.__add_medical_history(row)
        self.__add_physical_exams(row)
        self.__add_cognitive_screenings(row)
        self.__add_diagnosis(row)
        self.__add_specimen_assays(row)
        self.__add_imaging_assessments(row)

    def __init_visits(self, df: pd.DataFrame):
        """Adds visit nodes to the graph"""
        visits = df['visit_id'].unique()

        self.__add_visit_specifications(set(visits))

        subject_visits = df.groupby('subject_label')
        for subject_label, group_df in subject_visits:
            prev_visit = None
        
            # iterate the group_df rows
            for index, visit_row in group_df.iterrows():
                current_visit_uri = self.__add_subject_visit_uri(visit_row)
                # add temporal relations among visits of the same subject
                if prev_visit:
                    self.graph.add_edge(
                        prev_visit, 
                        current_visit_uri, 
                        **{"subject": prev_visit, 
                           "object": current_visit_uri, 
                           "predicate": 'biolink:precedes', 
                           "relation": self.global_map['precedes']
                           }
                    )
                prev_visit = current_visit_uri

    def __add_visit_specifications(self, visits: set):
        
        for visit in visits:
            visit_specification_node_id = self.__construct_node_id(f'visit_{visit}')
            self.graph.add_node(visit_specification_node_id, category='biolink:ClinicalEntity', name=f'Visit {visit}')
            self.__add_type_edge(visit_specification_node_id, self.global_map['visit specification'])
            
            # Visit code
            code_id = self.__construct_node_id(f'visit_{visit}_ID')
            id = 'IAO:0020000'
            self.graph.add_node(code_id, category='biolink:InformationContentEntity', name=f'visit_{visit}_ID', value=visit)
            self.__add_type_edge(code_id, self.global_map['Visit Code'])
            self.graph.add_edge(
                visit_specification_node_id, 
                code_id,  
                **{"subject": visit_specification_node_id, 
                   "object":  code_id, 
                   "predicate": 'biolink:has_part', 
                   "relation": self.global_map['has part']}
                )
    
    def __add_subject_visit_uri(self, row):
        subject_uri = self.get_subject_uri(row)
        subject_id = row[self.descriptor['subject']['id_col']]
        visit_id = row[self.descriptor['config']['visit_id_col_name']]
        visit_node_id = self.__construct_node_id(f'{subject_id}_{visit_id}')
        # subject participates in the visit
        self.__subject_participates_in_process(subject_uri, visit_node_id)
        
        # link visit to visit specification
        visit_specification_node_id = self.__construct_node_id(f'visit_{visit_id}')
        self.graph.add_edge(
            visit_specification_node_id,
            visit_node_id,             
            **{"subject": visit_specification_node_id, 
               "object": visit_node_id, 
               "predicate": 'biolink:related_to', 
               "relation": self.global_map['is about']}
        )

        return visit_node_id
    
    def __add_type_edge(self, node, type):
        """Adds an type edge to the graph for a node"""
        self.graph.add_edge(node, 
                    type, 
                    **{"subject": node, 
                        "object": type, 
                        "predicate": 'biolink:type', 
                        "relation": 'type'})
    
    def __read_subject(self, row):
        subject_id = row[self.descriptor['subject']['id_col']]
        subid = self.__construct_node_id(subject_id)

        human_class = 'NCBITaxon:9606'
        
        category = 'biolink:IndividualOrganism'
        self.graph.add_node(subid, category=category, name=subject_id)
        self.__add_type_edge(subid, human_class)

        female = 'PATO:0000383'
        male = 'PATO:0000384'
        sex = None
        has_phenotype = 'biolink:has_phenotype'
        has_quality = 'RO:0000086'
        
        if row['gender'] == 'male':
            sex = male
            
        elif row['gender'] == 'female':
            sex = female
            
        if sex:
            self.graph.add_edge(subid, 
                                sex, 
                                **{"subject": subid, 
                                   "object": sex, 
                                   "predicate": has_phenotype, 
                                   "relation": has_quality
                                   }
                                   )

        if 'handeness' in row:
            left = 'PATO:0002202'
            right = 'PATO:0002203'
            ambi = 'PATO:0002204'
            handedness = None

            if row['handedness'] == 'right':
                handedness = right

            elif row['handedness'] == 'left':
                handedness = left                

            if handedness:
                self.graph.add_edge(subid, 
                                    handedness, 
                                    **{"subject": subid, 
                                       "object": handedness, 
                                       "predicate": has_phenotype, 
                                       "relation": has_quality})

        return subid

    def __mention_phenotype(self, source: str, phenotype_entity: str):
        """
        Adds an edge to the graph indicating that the source entity mentions the phenotype entity.

        Parameters
        ----------
            source (str): The source entity.
            phenotype_entity (str): The phenotype entity.

        """
        if phenotype_entity is not None:
            self.graph.add_edge(source, 
                        phenotype_entity, 
                        **{"subject": source, 
                            "object": phenotype_entity, 
                            "predicate": 'biolink:mentions', 
                            "relation": self.global_map['mentions']})
            
    def __draw_conclusion(self, source: str, conclusion_object: str):
        """
        Adds nodes and edges to the graph to represent the process of drawing a conclusion based on data.

        Parameters
        ----------
        source : str
            The source of the data used to draw the conclusion.
        conclusion_object : str
            The object of the conclusion (e.g. phenotype, disease) that the conclusion is about.

        """
        if conclusion_object is not None:
            conclusion_process = self.__construct_node_uuid()
            self.graph.add_node(conclusion_process, category='biolink:ClinicalEntity', name=f'conclusion process')
            self.__add_type_edge(conclusion_process, self.global_map['drawing a conclusion based on data'])
            # the conclusion process has specified input the score datum
            self.__add_specified_input(conclusion_process, source)

            conclusion = self.__construct_node_uuid()
            self.graph.add_node(conclusion, category='biolink:ClinicalEntity', name=f'conclusion')
            self.__add_type_edge(conclusion, self.global_map['conclusion based on data'])
            # the conclusion is output of the conclusion process
            self.__add_specified_output(conclusion_process, conclusion)
            # the conclusion is about the object of the conclusion (phenotype, disease, etc.)
            self.graph.add_edge(conclusion,
                                conclusion_object,
                                **{"subject": conclusion,
                                    "object": conclusion_object,
                                    "predicate": 'biolink:is_about',
                                    "relation": self.global_map['is about']})
            
    def __add_medical_history(self, row):
        sub = self.__get_subject_node(row)
        encounter_uri = self.__get_planned_process_visit(row)
        mh = self.descriptor['medical_history']
        mh_label = self.__get_label('mh', row)
        mh_date = row[mh['date_col']]
        clinical_history_taking_process = self.__construct_node_id()
        self.graph.add_node(clinical_history_taking_process, category='biolink:ClinicalEntity', name=mh_label)
        type = self.global_map['OGMS:0000055']
        self.add_type_edge(clinical_history_taking_process, type)
        self.__add_has_part(encounter_uri, clinical_history_taking_process)

        for finding in mh['findings']:
            value = row[finding['col_name']]
            if not pd.isna(value):
                mappings = finding['value_mappings']
                phenotype_entity = self.__get_mapped_entity(mappings, value)
                self.__finding_mention_phenotype(sub, clinical_history_taking_process, self.global_map['clinical finding'], phenotype_entity, label=f'{mh_label}_finding')


    def __subject_participates_in_process(self, sub: str, process: str):
        self.graph.add_edge(sub, 
                    process, 
                    **{"subject": sub, 
                        "object": process, 
                        "predicate": 'biolink:participates_in',
                        "relation": 'RO:0000056'})
        self.graph.add_edge(process, 
                    sub, 
                    **{"subject": process, 
                        "object": sub, 
                        "predicate": 'biolink:has_participant', 
                        "relation": 'RO:0000057'})
        
    def __finding_mention_phenotype(self, sub: str, source_process: str, finding_type: str, phenotype_entity: str, label='finding'):
        if phenotype_entity is None:
            return
        
        finding = self.__construct_node_uuid()
        self.graph.add_node(finding, category='biolink:ClinicalEntity', name=label)
        self.__add_type_edge(finding, finding_type)

        # the finding is about the subject
        self.graph.add_edge(finding,
                        sub,
                            **{"subject": finding,
                            "object": sub,
                            "predicate": 'biolink:assesses',
                            "relation": self.global_map['is about']})

        self.graph.add_edge(source_process, 
                    finding, 
                    **{"subject": source_process, 
                        "object": finding,
                        "predicate": 'biolink:has_output', 
                        "relation": self.global_map['has_output']})
        self.__mention_phenotype(finding, phenotype_entity)
        return finding
    
    def __add_phyex_measurements(row, sub):
        pass

    def __add_physical_exams(self, row):
        sub = self.get_subject_uri(row)
        visit = self.__get_planned_process_visit(row)
        phyex_section = self.descriptor['physical_exam']
        findings = phyex_section['findings']

        physical_exam_process = self.__construct_node_uuid()
        phyex_label_field = phyex_section['id']
        phyex_label = self.__get_label(phyex_label_field, row)
        self.graph.add_node(physical_exam_process, category='biolink:ClinicalEntity', name=phyex_label)
        type = self.global_map['physical examination']
        self.__add_type_edge(physical_exam_process, type)

        finding_type = self.global_map['physical examination finding']
        for finding in findings:
            row_value = row[finding['col_name']]
            mappings = finding['value_mappings']
            phenotype_entity = self.__get_mapped_entity(mappings, row_value)
            self.__finding_mention_phenotype(sub, physical_exam_process, finding_type, phenotype_entity, label=f'{visit}_phyex_finding')
        
    
    def __add_cognitive_screenings(self, row):
        """
        Constructs nodes and edges for cognitive screening data.

        Parameters:
        row (pandas.core.series.Series): A row of data from the input file.

        Returns:
        None
        """
        if 'cognitive_screening' not in self.descriptor:
            return
        
        cs_section = self.descriptor['cognitive_screening']
        encounter_uri = self.__get_planned_process_visit(row)
        cs_label = self.__get_label('cognitive screening', row)
        cs_date = row[cs_section['date_col']]

        healthcare_process = self.__construct_node_uuid(element_hint='cs')
        self.graph.add_node(healthcare_process, category='biolink:ClinicalEntity', name=cs_label, date=cs_date)
        type = self.global_map['health care process assay']
        self.__add_type_edge(healthcare_process, type)
        self.__add_has_part(encounter_uri, healthcare_process)

        if 'scores' in cs_section:
            scores = cs_section['scores']
            for score in scores:
                # add test node and link to healthcare process
                test_class = score['test_class']
                test_node_id = self.__construct_node_uuid()
                self.graph.add_node(test_node_id, category='biolink:ClinicalEntity', name=cs_label)
                self.__add_type_edge(test_node_id, test_class)
                self.__add_has_part(healthcare_process, test_node_id)

                # add score node and link to test node
                score_val = row[score['col_name']]
                if not pd.isna(score_val):
                    mappings = score['value_mappings']
                    phenotype_entity = self.__get_mapped_entity(mappings, score_val)
                    score_id = self.__add_cs_score(score['type'], score_val, test_node_id)
                    # drawing a conclusion based on the data 
                    if phenotype_entity is not None:
                        conclusion_process = self.__construct_node_uuid()
                        self.graph.add_node(conclusion_process, category='biolink:ClinicalEntity', name=f'{cs_label} conclusion process')
                        self.__add_type_edge(conclusion_process, self.global_map['conclusion based on data'])
                        # the conclusion process has specified input the score datum
                        self.__add_specified_input(conclusion_process, score_id)

                        conclusion = self.__construct_node_uuid()
                        self.graph.add_node(conclusion, category='biolink:ClinicalEntity', name=f'{cs_label} conclusion')
                        self.__add_type_edge(conclusion, self.global_map['conclusion based on data'])
                        # the conclusion is output of the conclusion process
                        self.__add_specified_output(conclusion_process, conclusion)
                        # the conclusion is about the phenotype
                        self.graph.add_edge(conclusion,
                                            phenotype_entity,
                                            **{"subject": conclusion,
                                                "object": phenotype_entity,
                                                "predicate": 'biolink:is_about',
                                                "relation": self.global_map['is about']})
                        

    def __add_cs_score(self, score_type, score_value, test_node_id):
        """
        Adds a cognitive screening score to the graph.

        Parameters
        ----------
            score_type (str): The type of the score.
            score_value (float): The value of the score.
            test_node_id (str): The ID of the test node.

        Returns
        -------
            str: The ID of the score node.
        """
        if pd.isna(score_value):
            return
        score_node_id = self.__construct_node_uuid()
        self.graph.add_node(score_node_id, category='biolink:InformationContentEntity', name=score_type, value=score_value)
        self.__add_type_edge(score_node_id, score_type)
        self.graph.add_edge(
                    test_node_id, 
                    score_node_id, 
                    **{"subject": test_node_id, 
                        "object": score_node_id, 
                        "predicate": 'biolink:related_to', 
                        "relation": self.global_map['has_specified_output']
                        })
        return score_node_id
        
    def __add_diagnosis(self, row):
        """
        Adds a diagnostic process node to the graph, along with its associated encounter and phenotype entities.

        Parameters
        ----------
            row (pandas.Series): A row of a DataFrame containing the data for a single patient encounter.

        """
        if 'diagnosis' not in self.descriptor:
            return
        encounter_uri = self.__get_planned_process_visit(row)
        diagnosis = self.descriptor['diagnosis']
        label = self.__get_label('diagnosis', row)
        date = row[diagnosis['date_col']]
        diagnostic_process_node_id = self.__construct_node_uuid()
        self.graph.add_node(diagnostic_process_node_id, category='biolink:ClinicalEntity', name=label, date=date)
        type = self.global_map['diagnostic process']
        self.__add_type_edge(diagnostic_process_node_id, type)
        self.__add_has_part(encounter_uri, diagnostic_process_node_id)
        
        diagnosis_value = row[diagnosis['col_name']]
        if not pd.isna(diagnosis_value):
            diagnosis_type = self.global_map['diagnosis']
            mappings = diagnosis['value_mappings']
            phenotype_entity = self.__get_mapped_entity(mappings, diagnosis_value)
            sub = self.get_subject_uri(row)
            self.__finding_mention_phenotype(sub, diagnostic_process_node_id, diagnosis_type, phenotype_entity, label=f'{label}_finding')

    def __add_specimen_assays(self, row):
        if 'specimen_assays' not in self.descriptor:
            return
        
        # gather the specimen assays
        specimen_assays = self.descriptor['specimen_assays']
        encounter_uri = self.__get_planned_process_visit(row)
        subject = self.__get_subject_node(row)

        for assay in specimen_assays:
            # add the specimen collection process details
            specimen_date = row[assay['date_col']]
            specimen_collection_node = self.__construct_node_uuid()
            self.add_node(specimen_collection_node, category='biolink:ClinicalEntity', name=f'{assay["label"]} specimen collection', date=specimen_date)
            self.__add_has_part(encounter_uri, specimen_collection_node)
            
            # add the specimen details and connect as output of the specimen collection process
            specimen = assay['specimen']
            specimen_node = self.__construct_node_uuid()
            self.add_node(specimen_node, category='biolink:OrganismalEntity') # TODO could provide more specific subclass using ontology-biolink mappings 
            self.__add_type_edge(specimen_node, specimen)
            self.__add_specified_output(specimen_collection_node, specimen_node)

            # add all measurements 
            for measurement in assay['measurements']:
                measurement_value = row[measurement['col_name']]
                if not pd.isna(measurement_value):
                    assay_type = measurement['assay_type']
                    assay_date = row[measurement['date_col']]
                    assay_uri = self.__construct_node_uuid()            
                    self.graph.add_node(assay_uri, category='biolink:ClinicalEntity', date=assay_date)
                    self.__add_type_edge(assay_uri, assay_type)
                    # the specimen is the specified input of the assay
                    self.__add_specified_input(assay_uri, specimen_node)
                    # connect to the molecule entity via analite role
                    target_analyte_node = self.__construct_node_uuid()
                    self.add_node(target_analyte_node, category='biolink:Attribute')
                    self.__add_type_edge(target_analyte_node, self.global_map['analyte role'])
                    # the analyte role inheres in the target analyte
                    target_analyte = measurement['target_analyte']
                    self.__add_inheres_in(target_analyte, target_analyte_node)

                    measurement_datum_node = self.__construct_node_uuid()
                    self.add_node(measurement_datum_node, category='biolink:InformationContentEntity', name=f'measurement', value=measurement_value)
                    self.__add_type_edge(measurement_datum_node, self.global_map['measurement datum'])
                    # unit
                    if 'units' in measurement:
                        unit = measurement['units']
                        unit_node = self.__construct_node_uuid()
                        self.add_node(unit_node, category='biolink:Attribute', name=f'measurement unit')
                        self.__add_type_edge(unit_node, unit)
                    
                    if 'value_phenotype_mappings' in measurement:
                        phenotype_mappings = measurement['value_phenotype_mappings']
                        phenotype_entity = self.__get_mapped_entity(phenotype_mappings, measurement_value)
                        self.__draw_conclusion(measurement_datum_node, phenotype_entity)

    def __add_imaging_assessments(self, row):
        if 'imaging_assessments' not in self.descriptor:
            return
        
        # gather the imaging assessments
        imaging_assessments = self.descriptor['imaging_assessments']
        encounter_uri = self.__get_planned_process_visit(row)
        subject = self.__get_subject_node(row)

        for assessment in imaging_assessments:
            # add the imaging assessment process details
            assessment_id = row[assessment['id_col']]
            assessment_date = row[assessment['date_col']]
            imaging_assay_type = assessment['imaging_assay_type']
            source_session = assessment['source_session']
            assay_node = self.__construct_node_uuid()
            self.add_node(assay_node, category='biolink:Procedure', name=source_session)
            self.__add_type_edge(assay_node, imaging_assay_type)

            assessment_node = self.__construct_node_uuid()
            self.add_node(assessment_node, category='biolink:Dataset', name=assessment_id, date=assessment_date)
            self.__add_has_part(encounter_uri, assessment_node)
            # the assessment is about the assay
            self.graph.add_edge(
                assessment_node,
                assay_node,             
                **{"subject": assessment_node, 
                "object": assay_node, 
                "predicate": 'biolink:related_to', 
                "relation": self.global_map['is about']}
            )
            
            # add the measurement details and connect as output of the imaging assessment process
            for measurement in assessment['measurements']:
                measurement_value = row[measurement['col_name']]
                if not pd.isna(measurement_value):
                    measurement_type = measurement['measurement_type']
                    measured_entity = measurement['measured_entity']
                    
                    measurement_datum_node = self.__construct_node_uuid()
                    self.add_node(measurement_datum_node, category='biolink:InformationContentEntity', name=f'measurement', value=measurement_value)
                    self.__add_type_edge(measurement_datum_node, self.global_map['measurement datum'])
                    self.__add_has_part(assessment_node, measurement_datum_node)
                    # the measurement datum is about the measured entity
                    self.graph.add_edge(
                        measurement_datum_node,
                        measured_entity,             
                        **{"subject": measurement_datum_node, 
                            "object": measured_entity, 
                            "predicate": 'biolink:related_to_at_instance_level', 
                            "relation": self.global_map['is about']}
                    )
                    
                    # unit
                    if 'units' in measurement:
                        unit = measurement['units']
                        unit_node = self.__construct_node_uuid()
                        self.add_node(unit_node, category='biolink:Attribute', name=f'measurement unit')
                        self.__add_type_edge(unit_node, unit)
                    
                    if 'value_phenotype_mappings' in measurement:
                        phenotype_mappings = measurement['value_phenotype_mappings']
                        phenotype_entity = self.__get_mapped_entity(phenotype_mappings, measurement_value)
                        self.__draw_conclusion(measurement_datum_node, phenotype_entity)

    
    
            

    def __add_specified_input(self, specimen, assay_uri):
        self.graph.add_edge(assay_uri, 
            specimen, 
            **{"subject": assay_uri, 
            "object": specimen, 
            "predicate": 'biolink:related_to_at_instance_level', 
            "relation": self.global_map['has_specified_input']})
        self.graph.add_edge(specimen, 
            assay_uri, 
            **{"subject": specimen, 
            "object": assay_uri, 
            "predicate": 'biolink:related_to_at_instance_level', 
            "relation": self.global_map['is_specified_input_of']})

    def __add_specified_output(self, collection_process, specimen):
        self.graph.add_edge(specimen, 
                            collection_process, 
                            **{"subject": specimen, 
                               "object": collection_process, 
                               "predicate": 'biolink:related_to_at_instance_level', 
                               "relation": 'OBI:0000312'})
        self.graph.add_edge(collection_process, 
                            specimen, 
                            **{"subject": collection_process, 
                               "object": specimen, 
                               "predicate": 'biolink:related_to_at_instance_level', 
                               "relation": 'OBI:0000299'})

    def __add_role(self, specimen, role):
        self.graph.add_edge(specimen, 
                            role, 
                            **{"subject": specimen, 
                               "object": role, 
                               "predicate": 'biolink:related_to_at_instance_level', 
                               "relation": self.global_map['has role']})
        self.graph.add_edge(role, 
                            specimen, 
                            **{"subject": role, 
                               "object": specimen, 
                               "predicate": 'biolink:related_to_at_instance_level', 
                               "relation": self.global_map['role of']})

    def __add_realizes(self, process, realizable_entity):
        self.graph.add_edge(process, 
                            realizable_entity, 
                            **{"subject": process, 
                               "object": realizable_entity, 
                               "predicate": 'biolink:related_to_at_instance_level', 
                               "relation": self.global_map['realizes']})
        self.graph.add_edge(realizable_entity, 
                            process, 
                            **{"subject": realizable_entity, 
                               "object": process, 
                               "predicate": 'biolink:related_to_at_instance_level', 
                               "relation": self.global_map['is realized in']})

    def __add_inheres_in(self, dependent, bearer):
        """
        Adds an 'inheres in' relationship between the dependent and bearer nodes in the graph.

        Parameters
        ----------
        dependent (str): The dependent node in the relationship.
        bearer (str): The bearer node in the relationship.
        """
        self.graph.add_edge(dependent, bearer, **{"subject": dependent, "object": bearer, "predicate": 'biolink:related_to_at_instance_level', "relation": self.global_map['inheres in']})
        self.graph.add_edge(bearer, dependent, **{"subject": bearer, "object": dependent, "predicate": 'biolink:related_to_at_instance_level', "relation": self.global_map['bearer of']})

    def __add_has_part(self, whole, part):
        self.graph.add_edge(whole, part, **{"subject": whole, "object": part, "predicate": 'biolink:has_part', "relation": self.global_map['has part']})

    def __get_label(self, exp: str, row: str):
        """
        Returns the label for a given experiment and row of data.

        Parameters
        ----------
            exp (str): The name of the experiment.
            row (dict): A dictionary representing a row of data.

        Returns
        -------
            str: The label for the given experiment and row of data.
        """
        elabel = self.descriptor[exp]['id']
        if elabel in row:
            label = row[elabel]
        else:
            visit_id = row[self.descriptor['config']['visit_id_col_name']]
            subject_id = row[self.descriptor['subject']['id_col']]
            label = f'{subject_id}_{exp}_{visit_id}'
        return label

    def __get_subject_node(self, row):
            """
            Given a row of data, extract the subject ID and construct a node CURIE ID for it.

            Parameters
            ----------
                row (dataframe row): A list of values representing a row of data.

            Returns
            -------
                str: The node CURIE ID constructed from the subject label.
            """
            subject_label = row[self.descriptor['subject']['id_col']]
            return self.__construct_node_id(subject_label)
    
    def __get_visit_uri(self, row):
        """
        Constructs a CURIE for a visit node based on the visit ID column name in the input row.

        Parameters
        ----------
            row (pandas.Series): A row of data containing the visit ID column.

        Returns
        -------
            str: A CURIE for the visit node.
        """
        visit = row[self.descriptor['config']['visit_id_col_name']]
        return self.__construct_node_id(f'visit_{visit}')
    
    def __get_planned_process_visit(self, row):
        subject_label = row[self.descriptor['subject']['id_col']]
        visit_id = row[self.descriptor['config']['visit_id_col_name']]
        return self.__construct_node_id(f'{subject_label}_{visit_id}')
    
    def __construct_node_id(self, var):
        """
        Constructs a CURIE ID for a node in the dataset graph.

        Parameters
        ----------
            var (str): The string to include in the ID.

        Returns
        -------
            str: A CURIE ID string in the format 'PREFIX:var'.
        """
        return self.prefix + ':' + var

    def __construct_node_uuid(self, element_hint=''):
        """
        Constructs a UUID for a node in the dataset graph.

        Parameters
        ----------
            element_hint (str): An optional hint to include in the UUID.

        Returns
        -------
            str: A UUID string in the format 'PREFIX:{element_hint}{uuid4}'.
        """
        if element_hint != '':
            element_hint = f'{element_hint}_'
        return f'{self.prefix}:{element_hint}{uuid.uuid4()}'
            

    def __get_mapped_entity(self, mapping, value):
        """
        Given a dictionary of mappings and a value, return the mapped value.

        Parameters
        ----------
            mapping (dict): A dictionary of mappings.
            value (str): The value to map.

        Returns
        -------
            str: The mapped value.
        """
        mapping_type = mapping['type']
        mappings = mapping['mappings']
        if mapping_type == 'categorical':
            if value in mappings:
                return mappings[value]
            else:
                return None
        elif mapping_type == 'range':
            for mapping in mappings:
                if value >= mapping['min'] and value <= mapping['max']:
                    return mapping['value']
            return None
        elif mapping_type == 'cuttoff':
            for cutoff, category in mappings.items():
                if cutoff.startswith('>'):
                    cutoff_value = float(cutoff[1:])
                    if value > cutoff_value:
                        return category
                elif cutoff.startswith('<'):
                    cutoff_value = float(cutoff[1:])
                    if value < cutoff_value:
                        return category
                elif '-' in cutoff:
                    cutoff_range = cutoff.split('-')
                    cutoff_min = float(cutoff_range[0])
                    cutoff_max = float(cutoff_range[1])
                    if value >= cutoff_min and value <= cutoff_max:
                        return category
        return None
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a graph from a dataset.')
    parser.add_argument('input', type=str, help='The input file containing the data.')
    parser.add_argument('descriptor', type=str, help='The descriptor file containing the dataset description.', default='descriptor.yaml')
    args = parser.parse_args()

    descriptor = args.descriptor
    
    transformer = DemKGTransformer(descriptor)
    transformer.transform()
    transformer.save()
    