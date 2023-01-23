{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module attributes

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :template: class.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: {{ _('Submodules') }}
.. autosummary::
   :toctree:
   :template: base.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classdetails %}
{% if classes %}
{% for item in classes %}
.. rubric:: {{ _('Class') }} {{ item }}
.. autoclass:: {{ item }}
   :members:
   :show-inheritance:
   :inherited-members: str, list, set, frozenset, dict, Module
   :undoc-members:
   :special-members: __call__, __add__, __mul__, __div__, __floordiv__

{% endfor %}
{% endif %}
{% endblock %}

{% block functiondetails %}
{% if functions %}
.. rubric:: {{ _('Functions') }}
{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}
{% endif %}
{% endblock %}
