import dependencies.colors as color_scheme

analysis_options = [
    {"title": "Percentages of image samples by category", "href": "/"},
    {"title": "Minimal pixel color distribution", "href": "/min"},
    {"title": "Maximum pixel color distribution", "href": "/max"},
    {"title": "Mean pixel color distribution", "href": "/mean"},
    {"title": "Standard deviation pixel color distribution", "href": "/standard_deviation"},
    {"title": "Comparison of mean/standard deviation values", "href": "/mean_standard_deviation"},
    {"title": "Classification - Machine Learning", "href": "/ml_classification"},
    {"title": "Classification - Deep Learning accuracy", "href": "/dl_classification"},
    {"title": "Classification - Deep Learning matrix", "href": "/dl_classification_matrix"},
    {"title": "Sample images of all categories", "href": "/sample_images"},
    {"title": "Sample images of all categories (R-Channel)", "href": "/sample_images_channel"},
]

analysis_name = "COVID-19 Radiography Database"

analysis_description = "A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, " \
                       "Bangladesh in collaboration with medical doctors have created a database of chest X-ray " \
                       "images for COVID-19 positive cases, along with Normal and Viral Pneumonia images."

marker_labels = ["Healthy", "COVID-19", "Lung Opacity", "Viral Pneumonia"]

marker_colours = [color_scheme.color_200, color_scheme.color_400, color_scheme.color_600, color_scheme.color_800]

DESCRIPTOR_NORMAL = 0
DESCRIPTOR_COVID19 = 1
DESCRIPTOR_LUNG_OPACITY = 2
DESCRIPTOR_VIRAL_PNEUMONIA = 3

CLASSNAME_NORMAL = "Normal"
CLASSNAME_COVID19 = "COVID"
CLASSNAME_LUNG_OPACITY = "Lung_Opacity"
CLASSNAME_VIRAL_PNEUMONIA = "Viral_Pneumonia"
